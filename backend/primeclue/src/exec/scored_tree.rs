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
use crate::data::InputShape;
use crate::exec::node::Weighted;
use crate::exec::score::Score;
use crate::exec::tree::Tree;
use crate::serialization::{Deserializable, Serializable, Serializator};
use std::cmp::Ordering;
use std::cmp::Ordering::Greater;

#[derive(Debug, PartialEq, Clone)]
pub struct ScoredTree {
    score: Score,
    tree: Tree,
}

impl PartialOrd for ScoredTree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.score.partial_cmp(&other.score) {
            // tree with the same score but shorter is "better" so higher cmp
            None | Some(Ordering::Equal) => {
                other.tree.node_count().partial_cmp(&self.tree.node_count())
            }
            other => other,
        }
    }
}

impl ScoredTree {
    pub fn input_shape(&self) -> &InputShape {
        self.tree.input_shape()
    }

    pub fn score(&self) -> Score {
        self.score
    }

    pub fn set_score(&mut self, score: Score) {
        self.score = score
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    pub fn node_count(&self) -> usize {
        self.tree.node_count()
    }

    pub fn execute(&self, data: &DataView) -> Vec<f32> {
        self.tree.execute(data)
    }

    pub fn guess(&self, value: f32) -> Option<bool> {
        self.score.threshold().bool(value)
    }

    pub fn best_tree(trees: &[ScoredTree]) -> Option<&ScoredTree> {
        trees.iter().fold(None, |last, next| match last {
            None => Some(next),
            Some(last) => {
                if next.partial_cmp(last).unwrap() == Greater {
                    Some(next)
                } else {
                    Some(last)
                }
            }
        })
    }
}

impl Serializable for ScoredTree {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.score, &self.tree])
    }
}

impl Deserializable for ScoredTree {
    fn deserialize(s: &mut Serializator) -> Result<Self, String>
    where
        Self: Sized,
    {
        let score = Score::deserialize(s)?;
        let tree = Tree::deserialize(s)?;
        Ok(ScoredTree { score, tree })
    }
}

impl ScoredTree {
    pub fn new(tree: Tree, score: Score) -> Self {
        ScoredTree { score, tree }
    }

    pub fn execute_for_score(&self, data: &DataView) -> Option<Score> {
        self.tree.execute_for_score(data, self.score.class(), &self.score.objective())
    }

    pub fn get_start_node(&self) -> &Weighted {
        self.tree.get_start_node()
    }
}

#[cfg(test)]
mod test {
    use crate::data::outcome::Class;
    use crate::exec::score::Objective::Auc;
    use crate::exec::score::{Score, Threshold};
    use crate::exec::scored_tree::ScoredTree;
    use crate::exec::tree::test::{create_long_tree, create_short_tree};
    use crate::serialization::serializator::test::test_serialization;
    use std::cmp::Ordering;

    #[test]
    fn best_tree() {
        let trees = Vec::new();
        assert!(ScoredTree::best_tree(&trees).is_none());

        let t = create_long_tree();
        let s = Score::new(Auc, Class::new(0), 0.6, Threshold::new(0.0));
        let st = ScoredTree::new(t, s);
        let trees = vec![st];
        assert_eq!(ScoredTree::best_tree(&trees).unwrap(), &trees[0]);

        let t1 = create_long_tree();
        let s1 = Score::new(Auc, Class::new(0), 0.6, Threshold::new(0.0));
        let st1 = ScoredTree::new(t1, s1);
        let t2 = create_long_tree();
        let s2 = Score::new(Auc, Class::new(0), 0.8, Threshold::new(0.0));
        let st2 = ScoredTree::new(t2, s2);
        let trees = vec![st1, st2];
        assert_eq!(ScoredTree::best_tree(&trees).unwrap(), &trees[1]);
        let mut trees = trees.clone();
        trees.reverse();
        assert_eq!(ScoredTree::best_tree(&trees).unwrap(), &trees[0]);

        let t1 = create_short_tree();
        let s1 = Score::new(Auc, Class::new(0), 0.6, Threshold::new(0.0));
        let st1 = ScoredTree::new(t1, s1);
        let t2 = create_long_tree();
        let s2 = Score::new(Auc, Class::new(0), 0.6, Threshold::new(0.0));
        let st2 = ScoredTree::new(t2, s2);
        let trees = vec![st1, st2];
        assert_eq!(ScoredTree::best_tree(&trees).unwrap(), &trees[0]);
        let mut trees = trees.clone();
        trees.reverse();
        assert_eq!(ScoredTree::best_tree(&trees).unwrap(), &trees[1]);
    }

    #[test]
    fn cmp_score_greater() {
        let short = create_short_tree();
        let long = create_long_tree();
        let class = Class::new(0);
        let good = Score::new(Auc, class, 1.0, Threshold::new(0.0));
        let bad = Score::new(Auc, class, 0.6, Threshold::new(0.0));
        let bad_tree = ScoredTree::new(short, bad);
        let good_tree = ScoredTree::new(long, good);

        assert_eq!(good_tree.partial_cmp(&bad_tree), Some(Ordering::Greater))
    }

    #[test]
    fn cmp_score_less() {
        let short = create_short_tree();
        let long = create_long_tree();
        let class = Class::new(0);
        let good = Score::new(Auc, class, 1.0, Threshold::new(0.0));
        let bad = Score::new(Auc, class, 0.6, Threshold::new(0.0));
        let bad_tree = ScoredTree::new(short, bad);
        let good_tree = ScoredTree::new(long, good);

        assert_eq!(bad_tree.partial_cmp(&good_tree), Some(Ordering::Less))
    }

    #[test]
    fn cmp_size() {
        let short = create_short_tree();
        let long = create_long_tree();
        let class = Class::new(0);
        let s1 = Score::new(Auc, class, 1.0, Threshold::new(0.0));
        let s2 = s1.clone();
        let short_tree = ScoredTree::new(short, s1);
        let long_tree = ScoredTree::new(long, s2);

        assert_eq!(short_tree.partial_cmp(&long_tree), Some(Ordering::Greater))
    }

    #[test]
    fn cmp_same() {
        let t1 = create_short_tree();
        let t2 = t1.clone();
        let class = Class::new(0);
        let s1 = Score::new(Auc, class, 1.0, Threshold::new(0.0));
        let s2 = s1.clone();
        let st1 = ScoredTree::new(t1, s1);
        let st2 = ScoredTree::new(t2, s2);
        assert_eq!(st1.partial_cmp(&st2), Some(Ordering::Equal))
    }

    #[test]
    fn serialize() {
        let tree = create_long_tree();
        let score = Score::new(Auc, Class::new(2), 0.9, Threshold::new(1.0));
        let scored_tree = ScoredTree::new(tree, score);
        test_serialization(scored_tree);
    }
}
