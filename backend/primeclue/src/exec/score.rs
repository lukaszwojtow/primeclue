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

use crate::data::outcome::Class;
use crate::data::Outcome;
use crate::serialization::{Deserializable, Serializable, Serializator};
use core::fmt;
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;

/// An enum used to gauge a classifier goodness. Training is performed to maximize this
/// value.
/// * `Cost` - use cost function with reward and penalty for correct / incorrect predictions respectively
/// * `AUC` - use area under curve
/// * `Accuracy` - use simple accuracy
#[derive(Debug, PartialEq, Copy, Clone, serde::Deserialize, serde::Serialize)]
pub enum Objective {
    Cost,
    Auc,
    Accuracy,
}

impl Objective {
    pub fn threshold(&self, outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
        match self {
            Objective::Cost => cost_threshold(outcomes, class),
            Objective::Auc => auc_threshold(outcomes, class),
            Objective::Accuracy => accuracy_threshold(outcomes, class),
        }
    }

    pub fn score(
        &self,
        threshold: Threshold,
        outcomes: &[(f32, Outcome)],
        class: Class,
    ) -> Score {
        let value = match self {
            Objective::Auc => calculate_auc(outcomes, class),
            Objective::Accuracy => calculate_accuracy(threshold, outcomes, class),
            Objective::Cost => calculate_cost(threshold, outcomes, class),
        };
        Score { objective: *self, class, value, threshold }
    }
}

impl Serializable for Objective {
    fn serialize(&self, s: &mut Serializator) {
        let var = match self {
            Objective::Cost => "Cost",
            Objective::Auc => "AUC",
            Objective::Accuracy => "Accuracy",
        };
        s.add_str(var);
    }
}

impl Deserializable for Objective {
    fn deserialize(s: &mut Serializator) -> Result<Objective, String> {
        let t = s.next_token()?;
        match t.as_ref() {
            "Cost" => Ok(Objective::Cost),
            "AUC" => Ok(Objective::Auc),
            "Accuracy" => Ok(Objective::Accuracy),
            _ => Err(format!("Invalid token for ScoreType: {}", t)),
        }
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Objective::Cost => "Cost",
            Objective::Auc => "AUC",
            Objective::Accuracy => "Accuracy",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Threshold {
    value: f32,
}

impl Threshold {
    pub fn bool(self, f: f32) -> Option<bool> {
        if !f.is_finite() {
            None
        } else {
            Some(f >= self.value)
        }
    }

    pub fn new(f: f32) -> Self {
        Threshold { value: f }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

impl Serializable for Threshold {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.value)
    }
}

impl Deserializable for Threshold {
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        Ok(Threshold { value: f32::deserialize(s)? })
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Score {
    objective: Objective,
    class: Class,
    value: f32,
    threshold: Threshold,
}

impl Score {
    #[must_use]
    pub fn new(objective: Objective, class: Class, value: f32, threshold: Threshold) -> Score {
        Score { objective, class, value, threshold }
    }

    #[must_use]
    pub fn threshold(&self) -> Threshold {
        self.threshold
    }

    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    #[must_use]
    pub fn class(&self) -> Class {
        self.class
    }

    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    #[must_use]
    pub fn from(objective: Objective, class: Class, value: f32, threshold: Threshold) -> Self {
        Score { objective, class, value, threshold }
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.value.total_cmp(&other.value))
    }
}


impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            None => false,
            Some(ord) => ord == Equal,
        }
    }
}

impl Serializable for Score {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.objective, &self.class, &self.value, &self.threshold]);
    }
}

impl Deserializable for Score {
    fn deserialize(s: &mut Serializator) -> Result<Score, String> {
        let objective = Objective::deserialize(s)?;
        let class = Class::deserialize(s)?;
        let value = f32::deserialize(s)?;
        let threshold = Threshold::deserialize(s)?;
        Ok(Score { objective, class, value, threshold })
    }
}

#[must_use]
pub fn calculate_auc(outcomes: &[(f32, Outcome)], class: Class) -> f32 {
    let mut incorrect_count = 0_usize;
    let mut correct_count = 0_usize;
    let mut total_incorrect = 0_usize;
    for (_, outcome) in outcomes {
        if outcome.class() == class {
            correct_count += 1;
            total_incorrect += incorrect_count;
        } else {
            incorrect_count += 1;
        }
    }
    total_incorrect as f32 / (correct_count * incorrect_count) as f32
}

#[must_use]
fn calculate_accuracy(threshold: Threshold, outcomes: &[(f32, Outcome)], class: Class) -> f32 {
    let mut correct = 0;
    let mut total = 0;
    for (guess, outcome) in outcomes {
        if let Some(guess_bool) = threshold.bool(*guess) {
            total += 1;
            if (outcome.class() == class && guess_bool)
                || (outcome.class() != class && !guess_bool)
            {
                correct += 1;
            }
        }
    }
    correct as f32 / total as f32
}

#[must_use]
fn calculate_cost(threshold: Threshold, outcomes: &[(f32, Outcome)], class: Class) -> f32 {
    let mut cost = 0.0;
    for (guess, outcome) in outcomes {
        if let Some(b) = threshold.bool(*guess) {
            cost += outcome.calculate_cost(b, class);
        }
    }
    cost
}

#[must_use]
pub fn auc_threshold(outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
    let none_class_count =
        outcomes.iter().map(|(_, o)| o.class()).filter(|c| c != &class).count();
    if none_class_count == 0 {
        Threshold::new(outcomes[0].0)
    } else if none_class_count == outcomes.len() {
        Threshold::new(2.0 * outcomes[outcomes.len() - 1].0.abs())
    } else {
        Threshold::new(outcomes[none_class_count].0)
    }
}

#[must_use]
fn cost_threshold(outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
    let mut false_list = Vec::with_capacity(outcomes.len());
    let mut false_cost = 0.0;
    for (guess, outcome) in outcomes {
        let false_reward = outcome.calculate_cost(false, class);
        false_list.push((*guess, *outcome, false_cost));
        false_cost += false_reward;
    }
    false_list.reverse();
    let mut true_cost = 0.0;
    let mut cost_list = Vec::with_capacity(false_list.len());
    for (guess, outcome, false_reward) in false_list {
        let true_reward = outcome.calculate_cost(true, class);
        true_cost += true_reward;
        cost_list.push((guess, outcome, false_reward + true_cost));
    }
    cost_list.sort_by(|(_, _, cost1), (_, _, cost2)| cost1.total_cmp(cost2));
    Threshold::new(cost_list.last().unwrap().0)
}

#[must_use]
fn accuracy_threshold(outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
    let mut incorrect_list = Vec::with_capacity(outcomes.len());
    let mut incorrect_count = 0;
    for (guess, outcome) in outcomes {
        incorrect_list.push((*guess, *outcome, incorrect_count));
        if outcome.class() != class {
            incorrect_count += 1;
        }
    }
    let mut correct_count = 0;
    incorrect_list.reverse();
    let mut accuracy_list = Vec::with_capacity(incorrect_list.len());
    for (guess, outcome, incorrect_count) in incorrect_list {
        if outcome.class() == class {
            correct_count += 1;
        }
        accuracy_list.push((guess, outcome, incorrect_count + correct_count));
    }
    accuracy_list.sort_by(|(_, _, count1), (_, _, count2)| count1.cmp(count2));
    Threshold::new(accuracy_list.last().unwrap().0)
}

#[cfg(test)]
mod test {
    use crate::data::outcome::Class;
    use crate::data::Outcome;
    use crate::exec::score::Objective::{Accuracy, Auc, Cost};
    use crate::exec::score::{
        accuracy_threshold, auc_threshold, calculate_accuracy, calculate_auc, calculate_cost,
        cost_threshold, Score, Threshold,
    };
    use crate::serialization::serializator::test::test_serialization;
    use std::cmp::Ordering::Equal;

    #[test]
    fn serialize_tree_score() {
        test_serialization(Accuracy);
        test_serialization(Score {
            objective: Cost,
            class: Class::from(true),
            value: 0.0,
            threshold: Threshold::new(1.0),
        });
        test_serialization(Score {
            objective: Auc,
            class: Class::from(false),
            value: 1.0,
            threshold: Threshold::new(0.0),
        });
        test_serialization(Score {
            objective: Accuracy,
            class: Class::new(3),
            value: -11.5,
            threshold: Threshold::new(-10.3),
        });
    }

    #[test]
    fn cmp_incompatible_score() {
        let class = Class::new(0);
        let s1 = Score { objective: Auc, class, value: 1.0, threshold: Threshold::new(0.0) };
        let s2 = Score { objective: Cost, class, value: 1.0, threshold: Threshold::new(0.0) };
        assert_eq!(s1.partial_cmp(&s2), None);
        assert_ne!(s1, s2);
    }

    #[test]
    fn cmp_close_score_small() {
        let class = Class::new(0);
        let s1 = Score { objective: Auc, class, value: 0.7000, threshold: Threshold::new(0.0) };
        let s2 = Score { objective: Auc, class, value: 0.7005, threshold: Threshold::new(0.0) };
        assert_eq!(s1.partial_cmp(&s2).unwrap(), Equal);
        assert_eq!(s1, s2);
    }

    #[test]
    fn cmp_close_score_big() {
        let class = Class::new(0);
        let s1 =
            Score { objective: Auc, class, value: 1_000_000.0, threshold: Threshold::new(0.0) };
        let s2 =
            Score { objective: Auc, class, value: 1_000_001.0, threshold: Threshold::new(0.0) };
        assert_eq!(s1.partial_cmp(&s2).unwrap(), Equal);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_auc_correctness() {
        let p = Class::new(1);
        let n = Class::new(0);
        let predictions = vec![
            (0.01, Outcome::new(p, 1.0, -1.0)),
            (0.04, Outcome::new(p, 1.0, -1.0)),
            (0.11, Outcome::new(n, 1.0, -1.0)),
            (0.12, Outcome::new(p, 1.0, -1.0)),
            (0.15, Outcome::new(p, 1.0, -1.0)),
            (0.19, Outcome::new(p, 1.0, -1.0)),
            (0.22, Outcome::new(n, 1.0, -1.0)),
            (0.23, Outcome::new(n, 1.0, -1.0)),
            (0.31, Outcome::new(p, 1.0, -1.0)),
            (0.33, Outcome::new(n, 1.0, -1.0)),
            (0.39, Outcome::new(p, 1.0, -1.0)),
            (0.42, Outcome::new(n, 1.0, -1.0)),
            (0.43, Outcome::new(p, 1.0, -1.0)),
            (0.49, Outcome::new(n, 1.0, -1.0)),
            (0.51, Outcome::new(n, 1.0, -1.0)),
            (0.55, Outcome::new(n, 1.0, -1.0)),
            (0.60, Outcome::new(p, 1.0, -1.0)),
            (0.70, Outcome::new(n, 1.0, -1.0)),
            (0.80, Outcome::new(p, 1.0, -1.0)),
            (0.90, Outcome::new(n, 1.0, -1.0)),
        ];
        let auc = calculate_auc(&predictions, n);
        assert!((auc - 0.68).abs() < f32::EPSILON);
    }

    #[test]
    fn test_auc_threshold() {
        // Classification uses >= comparison
        // So if threshold is 3.14 then everything that is greater or equal 3.14 will be classified as true
        // Outcomes must be sorted in ascending order of first part of tuple
        let outcomes = vec![
            (-10.0, Outcome::new(Class::from(false), 1.0, -1.0)),
            (-3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = auc_threshold(&outcomes, Class::from(true)); // if false_count == 1 then return 2nd value
        assert!((t.value() - -3.0).abs() < f32::EPSILON);

        let outcomes = vec![
            (-10.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = auc_threshold(&outcomes, Class::from(false)); // if none false then return 2 * abs(last value)
        assert!((t.value() - 2.0).abs() < f32::EPSILON);

        let outcomes = vec![
            (1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (10.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = auc_threshold(&outcomes, Class::from(false)); // if none false then return double last value
        assert!((t.value() - 20.0).abs() < f32::EPSILON);

        let outcomes = vec![
            (-10.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = auc_threshold(&outcomes, Class::from(true)); // if none false then return first value
        assert!((t.value() - -10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cost_threshold() {
        let class = Class::new(1);
        for _ in 0..1_000 {
            let outcomes = get_biased_outcomes();
            let slow_threshold = naive_cost_threshold(&outcomes, class);
            let fast_threshold = cost_threshold(&outcomes, class);
            assert!((slow_threshold.value - fast_threshold.value).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_accuracy_threshold() {
        let class = Class::new(1);
        for _ in 0..1_000 {
            let outcomes = get_biased_outcomes();
            let slow_threshold = naive_accuracy_threshold(&outcomes, class);
            let fast_threshold = accuracy_threshold(&outcomes, class);
            assert!((slow_threshold.value - fast_threshold.value).abs() < f32::EPSILON);
        }
    }

    fn naive_cost_threshold(outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
        let mut max_score = 0.0;
        let mut threshold = Threshold::new(0.0);
        for (g, _) in outcomes {
            let t = Threshold::new(*g);
            let score = calculate_cost(t, outcomes, class);
            if score > max_score {
                threshold = t;
                max_score = score;
            }
        }
        threshold
    }

    fn naive_accuracy_threshold(outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
        let mut max_score = 0.0;
        let mut threshold = Threshold::new(0.0);
        for (g, _) in outcomes {
            let t = Threshold::new(*g);
            let score = calculate_accuracy(t, outcomes, class);
            if score > max_score {
                threshold = t;
                max_score = score;
            }
        }
        threshold
    }

    fn get_biased_outcomes() -> Vec<(f32, Outcome)> {
        use crate::contrand::GET_RNG;
        use rand::Rng;

        let mut rng = GET_RNG();
        let mut outcomes: Vec<(f32, Outcome)> = vec![];
        for _ in 0..50 {
            let guess = rng.gen_range(0.0..0.7);
            let reward = rng.gen_range(0.0..1.0);
            let penalty = rng.gen_range(-1.0..0.0);
            outcomes.push((guess, Outcome::new(Class::new(0), reward, penalty)));
        }
        for _ in 0..50 {
            let guess = rng.gen_range(0.3..1.0);
            let reward = rng.gen_range(0.0..1.0);
            let penalty = rng.gen_range(-1.0..0.0);
            outcomes.push((guess, Outcome::new(Class::new(1), reward, penalty)));
        }
        outcomes.sort_by(|(g1, _), (g2, _)| g1.total_cmp(g2));
        outcomes
    }
}
