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
use crate::error::PrimeclueErr;
use crate::exec::score::calculate_auc;
use crate::exec::scored_tree::ScoredTree;
use crate::serialization::{Deserializable, Serializable, Serializator};
use serde::Serialize;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct ClassifierScore {
    pub auc: f32,
    pub accuracy: f32,
    pub cost: f32,
    pub label_count_map: HashMap<String, usize>,
    pub label_accuracy_map: HashMap<String, f32>,
}

/// A structure containing a classifier trained via [`TrainingGroup`]
#[derive(Debug, PartialEq)]
pub struct Classifier {
    classes: HashMap<Class, String>,
    trees: Vec<ScoredTree>,
}

impl Classifier {
    pub fn new(
        classes: HashMap<Class, String>,
        trees: Vec<ScoredTree>,
    ) -> Result<Self, PrimeclueErr> {
        if classes.is_empty() {
            PrimeclueErr::result("Class map is empty".to_string())
        } else if trees.is_empty() {
            PrimeclueErr::result("Tree vectors is empty".to_string())
        } else if classes.len() != trees.len() {
            PrimeclueErr::result(format!(
                "Class / Tree vectors lengths don't match: {} vs {}",
                classes.len(),
                trees.len()
            ))
        } else {
            Ok(Classifier { classes, trees })
        }
    }

    pub fn get_classes(&self) -> &HashMap<Class, String> {
        &self.classes
    }

    pub fn input_shape(&self) -> &InputShape {
        self.trees[0].input_shape()
    }

    pub fn average_score(&self) -> Option<f32> {
        let mut val = 0.0;
        for tree in &self.trees {
            val += tree.score().value();
        }
        Some(val / self.trees.len() as f32)
    }

    pub fn save(&self, path: &Path, name: &str) -> Result<usize, PrimeclueErr> {
        let mut ser = Serializator::new();
        self.serialize(&mut ser);
        ser.save(path, format!("{}.ssd", name).as_str())
    }

    pub fn node_count(&self) -> usize {
        self.trees.iter().map(|t| t.node_count()).sum()
    }

    pub fn sorted_trees(&self) -> Vec<&ScoredTree> {
        let mut scores = self.trees.iter().collect::<Vec<_>>();
        scores.sort_unstable_by(|&t1, &t2| t1.partial_cmp(t2).unwrap_or(Equal));
        scores
    }

    pub fn classify(&self, data: &DataView) -> Vec<&str> {
        let trees = self.sorted_trees();
        let mut responses = vec![""; data.cells().get(0, 0).len()];
        for tree in &trees {
            let values = tree.execute(data);
            let class_string = self.classes.get(&tree.score().class()).unwrap();
            println!("Score for class {}: {}", class_string, tree.score().value());
            Classifier::show_adjusted_score(&values, class_string, data, tree.score().class());
            for (value, response) in values.iter().zip(responses.iter_mut()) {
                if let Some(guess) = tree.guess(*value) {
                    if guess {
                        *response = class_string;
                    }
                }
            }
        }
        // not allow empty
        // if let Some(tree) = trees.last() {
        //     // let class_string = self.classes.get(&tree.score().class()).unwrap();
        //     let class_string = self.classes.get(&Class::new(0)).unwrap();
        //     responses.iter_mut().for_each(|response| {
        //         if response == &"" {
        //             *response = class_string;
        //         }
        //     })
        // }
        // end not allow empty
        responses
    }

    fn show_adjusted_score(values: &[f32], label: &str, data: &DataView, class: Class) {
        let mut guesses =
            values.iter().copied().zip(data.outcomes().iter().copied()).collect::<Vec<_>>();
        guesses
            .sort_unstable_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap_or(Equal).reverse());
        guesses.truncate(data.class_items_count(class).unwrap());
        let score =
            guesses.iter().map(|(_, outcome)| outcome.calculate_cost(true, class)).sum::<f32>();
        println!("Adjusted score for {}, guess count {}: {}", label, guesses.len(), score);
    }

    fn execute_for_auc(&self, data: &DataView) -> Option<f32> {
        let mut sum_score = 0.0;
        for tree in &self.trees {
            sum_score += Classifier::calc_tree_auc(tree, data)?;
        }
        Some(sum_score / self.trees.len() as f32)
    }

    pub fn score(&self, data: &DataView) -> Option<ClassifierScore> {
        let auc = self.execute_for_auc(data)?;
        let predictions = self.classify(data);
        let mut label_count_map = HashMap::new();
        let mut label_accuracy_map = HashMap::new();
        let mut correct = 0;
        let mut total = 0;
        let mut reward = 0.0;
        let mut penalty = 0.0;
        for (prediction, outcome) in predictions.iter().zip(data.outcomes()) {
            if prediction.is_empty() {
                continue;
            }
            let label = (*prediction).to_owned();

            let class_count = label_count_map.remove(&label).unwrap_or(0) + 1;
            label_count_map.insert(label.clone(), class_count);

            if self.classes.get(&outcome.class()).unwrap() == &label {
                let accurate = label_accuracy_map.remove(&label).unwrap_or(0.0);
                label_accuracy_map.insert(label, accurate + 1.0);
            }

            total += 1;
            let expected = data.class_map().get(&outcome.class())?;
            if prediction == expected {
                correct += 1;
                reward += outcome.reward();
            } else {
                penalty += outcome.penalty();
            }
        }
        for (label, correct) in label_accuracy_map.iter_mut() {
            let count = label_count_map.get(label).unwrap();
            *correct /= *count as f32;
        }
        let accuracy = correct as f32 / total as f32;
        let cost = reward + penalty;
        Some(ClassifierScore { auc, accuracy, cost, label_count_map, label_accuracy_map })
    }

    fn calc_tree_auc(tree: &ScoredTree, data: &DataView) -> Option<f32> {
        let values = tree.execute(data);
        if values.iter().any(|v| !v.is_finite()) {
            None
        } else {
            let mut outcomes =
                values.into_iter().zip(data.outcomes().iter().copied()).collect::<Vec<_>>();
            outcomes.sort_unstable_by(|(v1, _), (v2, _)| v1.partial_cmp(v2).unwrap());
            Some(calculate_auc(&outcomes, tree.score().class()))
        }
    }
}

impl Serializable for Classifier {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.classes);
        s.add(&self.trees)
    }
}

impl Deserializable for Classifier {
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        let classes = HashMap::deserialize(s)?;
        let trees = Vec::deserialize(s)?;
        Ok(Classifier { classes, trees })
    }
}

#[cfg(test)]
mod test {
    use crate::data::data_set::test::create_simple_data;
    use crate::data::outcome::Class;
    use crate::data::InputShape;
    use crate::exec::classifier::Classifier;
    use crate::exec::score::Objective::Auc;
    use crate::exec::score::{Objective, Score, Threshold};
    use crate::exec::scored_tree::ScoredTree;
    use crate::exec::training_group::TrainingGroup;
    use crate::exec::tree::Tree;
    use crate::serialization::serializator::test::test_serialization;
    use std::collections::HashMap;

    #[test]
    fn serialize_classifier() {
        for _ in 0..10 {
            let (d1, d2, _) = create_simple_data(100).shuffle().into_3_views_split();
            let mut training_group: TrainingGroup<'_, Objective> =
                TrainingGroup::new(d1, d2, &Auc, 5, &[]).unwrap();
            loop {
                training_group.next_generation();
                if let Ok(classifier) = training_group.classifier() {
                    test_serialization(classifier);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_empty_classifier() {
        let classes = HashMap::new();
        let r = Classifier::new(classes, vec![]);
        assert!(r.is_err());

        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "true".to_string());
        classes.insert(Class::new(1), "false".to_string());
        let r = Classifier::new(classes, vec![]);
        assert!(r.is_err());

        let t = Tree::new(&InputShape::new(1, 1), 3, &[], 0.5, 0.5);
        let classes = HashMap::new();
        let r = Classifier::new(
            classes,
            vec![ScoredTree::new(t, Score::new(Auc, Class::new(0), 0.9, Threshold::new(0.0)))],
        );
        assert!(r.is_err());
    }

    #[test]
    fn invalid_class_count() {
        let mut trees = vec![];
        for i in 0..3 {
            let t = Tree::new(&InputShape::new(1, 1), 3, &[], 0.5, 0.5);
            trees.push(ScoredTree::new(
                t,
                Score::new(Auc, Class::new(i), 0.9, Threshold::new(0.0)),
            ))
        }
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "true".to_string());
        classes.insert(Class::new(1), "false".to_string());
        let r = Classifier::new(classes, trees);
        assert!(r.is_err());

        let mut trees = vec![];
        for i in 0..2 {
            let t = Tree::new(&InputShape::new(1, 1), 3, &[], 0.5, 0.5);
            trees.push(ScoredTree::new(
                t,
                Score::new(Auc, Class::new(i), 0.9, Threshold::new(0.0)),
            ))
        }
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "true".to_string());
        classes.insert(Class::new(1), "false".to_string());
        classes.insert(Class::new(2), "null".to_string());
        let r = Classifier::new(classes, trees);
        assert!(r.is_err());
    }
}
