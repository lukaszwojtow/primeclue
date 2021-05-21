// SPDX-License-Identifier: AGPL-3.0
/*
   Primeclue: Machine Learning and Data Mining
   Copyright (C) 2020 Łukasz Wojtów

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

use crate::data::data_set::DataView;
use crate::data::outcome::Class;
use crate::data::InputShape;
use crate::exec::functions::TWO_ARG_FUNCTIONS;
use crate::exec::score::{AsObjective, Score};
use crate::exec::scored_tree::ScoredTree;
use crate::exec::tree::Tree;
use crate::rand::GET_RNG;
use rand::prelude::SliceRandom;
use rand::seq::IteratorRandom;
use rand::Rng;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::fmt::{Debug, Error, Formatter};
use std::marker::PhantomData;
use std::mem::replace;

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct GroupId(u64);

pub struct ClassTraining<'o, T: AsObjective> {
    next_id: GroupId,
    objective: &'o T,
    size: usize,
    node_limit: usize,
    forbidden_cols: Vec<usize>,
    best_tree: Option<ScoredTree>,
    class: Class,
    groups: HashMap<GroupId, ClassGroup<T>>,
}

impl<T: AsObjective> Debug for ClassTraining<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:?}", self.class)
    }
}

impl<'o, T: AsObjective> ClassTraining<'o, T> {
    #[must_use]
    pub fn new(size: usize, forbidden_cols: Vec<usize>, objective: &'o T, class: Class) -> Self {
        let groups = HashMap::new();
        ClassTraining {
            next_id: GroupId(1),
            size,
            forbidden_cols,
            groups,
            node_limit: 5_000_000,
            best_tree: None,
            objective,
            class,
        }
    }

    pub fn class(&self) -> &Class {
        &self.class
    }

    pub fn training_score(&self) -> Option<f32> {
        self.best_tree.as_ref().map(|t| t.score().value())
    }

    #[must_use]
    pub fn best_tree(&self) -> Option<&ScoredTree> {
        self.best_tree.as_ref()
    }

    pub fn next_generation(&mut self, training_data: &DataView, verification_data: &DataView) {
        self.fill_up(training_data.input_shape());
        let objective = &self.objective; //.clone();
        let class = self.class;
        let length = self.size;
        let forbidden_cols = &self.forbidden_cols;
        self.groups.par_iter_mut().for_each(|(_, group)| {
            // TODO
            group.breed(forbidden_cols, length);
            group.execute_and_score(objective, training_data, class);
            group.remove_weak_trees(length);
        });
        self.remove_empty_groups();
        self.select_best(verification_data);
        self.keep_node_limit();
        self.groups.shrink_to_fit();
    }

    fn remove_empty_groups(&mut self) {
        self.groups.retain(|_, p| !p.scored.is_empty());
    }

    fn keep_node_limit(&mut self) {
        let mut sizes =
            self.groups.values().map(|p| (p.id, p.nodes_count())).collect::<Vec<_>>();
        let sum = sizes.iter().map(|(_, s)| s).sum::<usize>();
        if sum > self.node_limit {
            sizes.sort_by(|(_, s1), (_, s2)| s1.cmp(s2));
            let mut so_far = 0;
            for (id, size) in sizes {
                if so_far + size > self.node_limit {
                    self.groups.remove(&id);
                } else {
                    so_far += size;
                }
            }
        }
    }

    fn fill_up(&mut self, input_shape: &InputShape) {
        while self.groups.len() < self.size * 2 {
            let id = self.next_id;
            self.next_id.0 += 1;
            let group = generate_group(self, input_shape, id, &self.forbidden_cols, 3);
            self.groups.insert(group.id, group);
        }
    }

    fn select_best(&mut self, data: &DataView) {
        let mut sorted_scores = self.sorted_by_score(data);
        self.assign_best_tree(&sorted_scores);
        self.remove_bad_groups(&mut sorted_scores);
    }

    fn remove_bad_groups(&mut self, sorted_scores: &mut Vec<(GroupId, Score)>) {
        if self.groups.len() <= self.size {
            return;
        }
        let mut new_group_map = HashMap::with_capacity(self.size);
        for _ in 0..self.size {
            if !sorted_scores.is_empty() {
                let (first, _) = sorted_scores.remove(0);
                new_group_map.insert(first, self.groups.remove(&first).unwrap());
            }
        }
        self.groups = new_group_map;
    }

    fn assign_best_tree(&mut self, sorted_scores: &[(GroupId, Score)]) {
        if !sorted_scores.is_empty() {
            let mut best_now =
                ScoredTree::best_tree(&self.groups.get(&sorted_scores[0].0).unwrap().scored)
                    .unwrap()
                    .clone();
            let score_value = (sorted_scores[0].1.value() + best_now.score().value()) / 2.0;
            let score = Score::new(
                best_now.score().objective(),
                best_now.score().class(),
                score_value,
                best_now.score().threshold(),
            );
            best_now.set_score(score);
            if self.best_tree.is_none() || (&best_now > self.best_tree.as_ref().unwrap()) {
                self.best_tree = Some(best_now);
            }
        }
    }

    fn sorted_by_score(&self, data: &DataView) -> Vec<(GroupId, Score)> {
        let mut scores = Vec::with_capacity(self.groups.len());
        for g in self.groups.values() {
            if let Some(tree) = ScoredTree::best_tree(&g.scored) {
                if let Some(score) = tree.execute_for_score(data) {
                    scores.push((g.id, score))
                }
            }
        }
        scores.sort_unstable_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap());
        scores.reverse();
        scores
    }
}

pub struct ClassGroup<T: AsObjective> {
    id: GroupId,
    fresh: Vec<Tree>,
    scored: Vec<ScoredTree>,
    phantom: PhantomData<T>,
}

impl<T: AsObjective> Debug for ClassGroup<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.id.0)
    }
}

impl<T: AsObjective> ClassGroup<T> {
    fn create_joined(
        group_size: usize,
        existing: &HashMap<GroupId, ClassGroup<T>>,
        id: GroupId,
        forbidden_cols: &[usize],
    ) -> Option<Self> {
        let mut rng = GET_RNG();
        let tree1 = existing.values().choose(&mut rng)?.scored.iter().choose(&mut rng)?.tree();
        let tree2 = existing.values().choose(&mut rng)?.scored.iter().choose(&mut rng)?.tree();
        let tree = Tree::from_two(
            TWO_ARG_FUNCTIONS.choose(&mut rng).unwrap(),
            tree1.get_start_node().clone(),
            tree2.get_start_node().clone(),
            *tree1.input_shape(),
        );
        Some(ClassGroup::create_from_tree(group_size, id, tree, forbidden_cols))
    }

    fn create_random(
        group_size: usize,
        input_shape: &InputShape,
        id: GroupId,
        max_depth: usize,
        forbidden_cols: &[usize],
    ) -> Self {
        let mut rng = GET_RNG();
        let data_prob = rng.gen_range(0.01, 0.99);
        let branch_prob = rng.gen_range(0.01, 0.99);
        let tree = Tree::new(input_shape, max_depth, forbidden_cols, branch_prob, data_prob);
        ClassGroup::create_from_tree(group_size, id, tree, forbidden_cols)
    }

    fn create_from_tree(
        group_size: usize,
        id: GroupId,
        tree: Tree,
        forbidden_cols: &[usize],
    ) -> ClassGroup<T> {
        let mut trees = Vec::with_capacity(group_size);
        trees.push(tree);
        while trees.len() < group_size {
            let mut t = trees[0].clone();
            t.change_weights();
            t.mutate(forbidden_cols);
            trees.push(t);
        }
        ClassGroup { id, fresh: trees, scored: Vec::new(), phantom: PhantomData::default() }
    }

    fn breed(&mut self, forbidden_cols: &[usize], count: usize) {
        let mut rng = GET_RNG();
        while self.fresh.len() < count {
            if let Some(tree) = self.scored.choose(&mut rng).map(|t| t.tree()) {
                let mut child = tree.clone();
                child.mutate(forbidden_cols);
                self.fresh.push(child);

                let mut child = tree.clone();
                child.change_weights();
                self.fresh.push(child);

                let mut child = tree.clone();
                child.mutate(forbidden_cols);
                child.change_weights();
                self.fresh.push(child);
            }
        }
    }

    fn remove_weak_trees(&mut self, length: usize) {
        self.scored.sort_unstable_by(|t1, t2| t1.partial_cmp(&t2).unwrap_or(Equal));
        self.scored.reverse();
        self.scored.truncate(length);
    }

    fn execute_and_score(&mut self, objective: &T, data: &DataView, class: Class) {
        let len = self.fresh.len();
        let trees = replace(&mut self.fresh, Vec::with_capacity(len));
        for tree in trees {
            if let Some(score) = tree.execute_for_score::<T>(data, class, objective) {
                self.scored.push(ScoredTree::new(tree, score))
            }
        }
    }

    #[must_use]
    fn nodes_count(&self) -> usize {
        self.scored.iter().map(|t| t.node_count()).sum::<usize>()
            + self.fresh.iter().map(|t| t.node_count()).sum::<usize>()
    }
}

fn generate_group<T: AsObjective>(
    training: &ClassTraining<'_, T>,
    input_shape: &InputShape,
    id: GroupId,
    forbidden_cols: &[usize],
    max_depth: usize,
) -> ClassGroup<T> {
    let mut rng = GET_RNG();
    if !training.groups.is_empty() && rng.gen_bool(0.5) {
        if let Some(group) =
            ClassGroup::create_joined(training.size, &training.groups, id, forbidden_cols)
        {
            return group;
        }
    }
    ClassGroup::create_random(training.size, &input_shape, id, max_depth, forbidden_cols)
}
