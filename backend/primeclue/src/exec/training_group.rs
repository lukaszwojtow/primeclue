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
use crate::error::PrimeclueErr;
use crate::exec::class_training::ClassTraining;
use crate::exec::classifier::Classifier;
use crate::exec::score::{AsObjective, Score};
use crate::exec::scored_tree::ScoredTree;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::Serialize;
use std::mem::replace;

#[derive(Debug)]
pub struct TrainingGroup<'o, T: AsObjective> {
    generation: u32,
    training_data: DataView,
    verification_data: DataView,
    classes: Vec<ClassTraining<'o, T>>,
    objective: &'o T,
    thread_pool: ThreadPool,
}

impl<'o, T: AsObjective> TrainingGroup<'o, T> {
    /// Creates a new [`TrainingGroup`] that can be used to train a classifier through
    /// its [`next_generation`] method.
    ///
    /// # Arguments
    /// * `training_data` - [`DataView`] that represents training data view
    /// * `verification_data` - [`DataView`] that represents verification data view
    /// * `objective` - [`Objective`] that represents the measure to optimize for
    /// * `size` - size of a training group. Determines amount of RAM needed
    /// * `forbidden_cols` - indexes of data columns that should not be used as input
    pub fn new(
        training_data: DataView,
        verification_data: DataView,
        objective: &'o T,
        size: usize,
        forbidden_cols: &[usize],
    ) -> Result<Self, PrimeclueErr> {
        validate(&training_data, &verification_data)?;
        let classes = (0..training_data.class_count())
            .map(|class| {
                ClassTraining::new(
                    size,
                    forbidden_cols.to_vec(),
                    objective,
                    Class::new(class as u16),
                )
            })
            .collect();
        let num_threads = 64;
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| format!("Unable to build thread pool: {:?}", e))?;
        Ok(TrainingGroup {
            objective,
            generation: 0,
            training_data,
            verification_data,
            classes,
            thread_pool,
        })
    }

    /// Performs training for one generation
    pub fn next_generation(&mut self) {
        self.generation += 1;
        let training_data = &self.training_data;
        let verification_data = &self.verification_data;
        let mut classes = replace(&mut self.classes, vec![]);
        self.thread_pool.scope(|s| {
            for class in &mut classes {
                s.spawn(move |_| {
                    class.next_generation(training_data, verification_data);
                })
            }
        });
        self.classes = classes;
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn stats(&self) -> Option<Stats> {
        let mut node_count = 0;
        let mut training_score = 0.0;
        for class in &self.classes {
            let best_tree = class.best_tree()?;
            node_count += best_tree.node_count();
            training_score += class.training_score()?;
        }
        Some(Stats { generation: self.generation, node_count, training_score })
    }

    pub fn get_tree(&self, class: &Class) -> Option<&ScoredTree> {
        let class_training = self.classes.iter().find(|&c| c.class() == class)?;
        class_training.best_tree()
    }

    /// Get [`Classifier`] after training. [`Classifier`] can later be used for
    /// classification on unseen data.
    pub fn classifier(&self) -> Result<Classifier, PrimeclueErr> {
        let mut trees = Vec::new();
        for p in &self.classes {
            if let Some(t) = p.best_tree() {
                trees.push(t.clone());
            } else {
                return PrimeclueErr::result(format!(
                    "Unable to get classifier for class {}",
                    self.training_data.class_map().get(p.class()).unwrap()
                ));
            }
        }
        let classes = self.training_data.class_map().clone();
        Classifier::new(classes, trees).map_err(|e| {
            PrimeclueErr::from(format!("Unable to create a classifier: {}", e.to_string()))
        })
    }
}

fn validate(training_data: &DataView, verification_data: &DataView) -> Result<(), PrimeclueErr> {
    if training_data.cells().is_empty() {
        PrimeclueErr::result("Data training set is empty".to_string())
    } else if verification_data.cells().is_empty() {
        PrimeclueErr::result("Data verification set is empty".to_string())
    } else if verification_data.class_count() != training_data.class_count() {
        PrimeclueErr::result(format!(
            "Training and verification data differ in class count: {} vs {}",
            training_data.class_count(),
            verification_data.class_count()
        ))
    } else if verification_data.input_shape() != training_data.input_shape() {
        PrimeclueErr::result(format!(
            "Training and verification data differ in data size: {:?} vs {:?}",
            training_data.input_shape(),
            verification_data.input_shape()
        ))
    } else {
        Ok(())
    }
}

#[derive(Serialize, Debug, Copy, Clone)]
pub struct Stats {
    pub generation: u32,
    pub training_score: f32,
    pub node_count: usize,
}

#[derive(Serialize, Debug)]
pub struct ClassScore {
    class: String,
    score: Score,
}

#[cfg(test)]
mod test {
    use crate::data::data_set::test::create_simple_data;
    use crate::data::outcome::Class;
    use crate::exec::score::Objective::Auc;
    use crate::exec::training_group::TrainingGroup;

    #[test]
    fn test_generation() {
        let data = create_simple_data(100);
        let (training_data, verification_data, _) = data.shuffle().into_3_views_split();
        let mut training_group =
            TrainingGroup::new(training_data, verification_data, &Auc, 3, &Vec::new()).unwrap();
        training_group.next_generation();
        training_group.next_generation();
        training_group.next_generation();
        assert_eq!(training_group.generation(), 3)
    }

    #[test]
    fn test_get_tree() {
        let (training_data, verification_data) = create_simple_data(1_000).into_2_views_split();
        let mut training_group =
            TrainingGroup::new(training_data, verification_data, &Auc, 10, &Vec::new()).unwrap();

        for _ in 0..1_000 {
            training_group.next_generation();
            if let Some(_) = training_group.get_tree(&Class::new(1)) {
                return;
            }
        }
        panic!("get_tree(class) failed")
    }
}
