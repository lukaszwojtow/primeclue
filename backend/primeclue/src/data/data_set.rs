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
use crate::data::{Data, Input, InputShape, Outcome};
use crate::error::PrimeclueErr;
use crate::rand::GET_RNG;
use crate::serialization::{Deserializable, Serializable, Serializator};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::slice::Iter;

#[derive(Clone, PartialEq, Debug)]
pub struct Point {
    input: Input,
    outcome: Outcome,
}

#[derive(serde::Deserialize, Debug, Copy, Clone)]
pub struct Rewards {
    reward: f32,
    penalty: f32,
}

const DATA_FILE_NAME: &str = "data.ssd";

impl Point {
    #[must_use]
    pub fn new(input: Input, outcome: Outcome) -> Point {
        Point { input, outcome }
    }

    #[must_use]
    pub fn data(&self) -> (&Input, &Outcome) {
        (&self.input, &self.outcome)
    }
}

impl Serializable for Point {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.input, &self.outcome])
    }
}

impl Deserializable for Point {
    fn deserialize(s: &mut Serializator) -> Result<Point, String> {
        let input = Input::deserialize(s)?;
        let outcome = Outcome::deserialize(s)?;
        Ok(Point { input, outcome })
    }
}

/// A structure that represents a cut through all data points within [`DataSet`] for
/// each row/column coordinate.
/// This is done mainly for performance (CPU cache and such) as dealing with vectors
/// is usually faster than individual cells
#[derive(Debug, Clone)]
pub struct DataView {
    cells: Data<Vec<f32>>,
    outcomes: Vec<Outcome>,
    class_count: HashMap<Class, usize>,
    class_map: HashMap<Class, String>,
}

impl DataView {
    pub fn random_guess_cost(&self) -> f32 {
        let count = 1_000;
        let mut sum = 0.0;
        for _ in 0..count {
            sum += self.random_guess_cost_once();
        }
        sum / count as f32
    }

    fn random_guess_cost_once(&self) -> f32 {
        let mut cost = 0.0;
        let mut rng = GET_RNG();
        for outcome in &self.outcomes {
            let mut index = rng.gen_range(0, self.outcomes.len()) as i32;
            for (class, count) in &self.class_count {
                index -= *count as i32;
                if index <= 0 {
                    cost += outcome.calculate_cost(true, *class);
                    break;
                }
            }
        }
        cost
    }

    pub fn cost_range(&self) -> (f32, f32) {
        let mut reward = 0.0;
        let mut penalty = 0.0;
        for point in &self.outcomes {
            reward += point.reward();
            penalty += point.penalty();
        }
        (penalty, reward)
    }

    pub fn class_count(&self) -> usize {
        self.class_count.len()
    }

    pub fn input_shape(&self) -> &InputShape {
        self.cells.input_shape()
    }

    pub fn cells(&self) -> &Data<Vec<f32>> {
        &self.cells
    }

    pub fn outcomes(&self) -> &Vec<Outcome> {
        &self.outcomes
    }

    pub fn class_map(&self) -> &HashMap<Class, String> {
        &self.class_map
    }
}

#[derive(PartialEq, Debug, Default, Clone)]
pub struct DataSet {
    points: Vec<Point>,
    classes: HashMap<Class, String>,
}

impl DataSet {
    #[must_use]
    pub fn new(classes: HashMap<Class, String>) -> DataSet {
        DataSet { points: vec![], classes }
    }

    #[must_use]
    pub fn cost_range(&self) -> (f32, f32) {
        let mut max = 0.0;
        let mut min = 0.0;
        self.points.iter().for_each(|ds| {
            max += ds.outcome.reward();
            min += ds.outcome.penalty();
        });
        (min, max)
    }

    /// Filters [DataSet] to contains only points matching the predicate
    /// It leaves original class map untouched so it may contain class labels
    /// for classes no longer present.
    #[must_use]
    pub fn filter<F>(self, predicate: F) -> Self
    where
        F: Fn(&Point) -> bool,
    {
        let mut new = DataSet::new(self.classes.clone()); // TODO consider checking if all classes are present in `new`
        for point in self.points {
            if predicate(&point) {
                new.add_data_point(point).unwrap();
            }
        }
        new
    }

    #[must_use]
    pub fn into_view(self) -> DataView {
        // TODO change to Option, None if empty
        let mut cells = Data::new();
        let input_shape = self.input_shape();
        for row in 0..input_shape.rows() {
            let mut row_data = Vec::with_capacity(input_shape.columns());
            for column in 0..input_shape.columns() {
                let mut data = Vec::new();
                for set in self.iter() {
                    data.push(set.input.get(row, column));
                }
                row_data.push(data);
            }
            cells.add_row(row_data).unwrap();
        }

        let mut outcomes = Vec::with_capacity(self.len());
        let mut class_count = HashMap::new();
        for set in self.points {
            let outcome = set.outcome;
            outcomes.push(outcome);
            let count = class_count.remove(&outcome.class()).unwrap_or(0);
            class_count.insert(outcome.class(), count + 1);
        }
        DataView { outcomes, cells, class_count, class_map: self.classes.clone() }
    }

    pub fn add_data_point(&mut self, point: Point) -> Result<(), String> {
        let (input, _) = point.data();
        if !self.points.is_empty() {
            if input.input_shape() != self.input_shape() {
                return Err(format!(
                    "Invalid input size, expecting {}, got: {}",
                    self.input_shape(),
                    input.input_shape()
                ));
            }
            if !self.classes.contains_key(&point.outcome.class()) {
                return Err(format!(
                    "Class {:?} not found in DataSet's classes",
                    point.outcome.class()
                ));
            }
        }

        self.points.push(point);
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.len() == 0
    }

    #[must_use]
    pub fn input_shape(&self) -> &InputShape {
        self.points[0].input.input_shape() // TODO check for empty points
    }

    #[must_use]
    pub fn iter(&self) -> Iter<'_, Point> {
        self.points.iter()
    }

    /// Splits [`DataSet`] into three equal [`DataView`].
    /// No attempt is made to ensure equal class count in each [`DataView`]
    pub fn into_3_views_split(self) -> (DataView, DataView, DataView) {
        let (s1, s2, s3) = self.split3();
        (s1.into_view(), s2.into_view(), s3.into_view())
    }

    pub fn split_with_test_data_marker<P>(self, predicate: P) -> (DataView, DataView, DataView)
    where
        P: Fn(&Point) -> bool,
    {
        let mut training_set = DataSet::new(self.classes.clone());
        let mut verification_set = DataSet::new(self.classes.clone());
        let mut testing_set = DataSet::new(self.classes.clone());
        let mut rng = GET_RNG();
        for point in self.points {
            if predicate(&point) {
                testing_set.add_data_point(point).unwrap();
            } else if rng.gen_bool(0.5) {
                training_set.add_data_point(point).unwrap();
            } else {
                verification_set.add_data_point(point).unwrap();
            }
        }
        (training_set.into_view(), verification_set.into_view(), testing_set.into_view())
    }

    /// Splits [`DataSet`] into two equal [`DataView`].
    /// No attempt is made to ensure equal class count in each [`DataView`]
    pub fn into_2_views_split(self) -> (DataView, DataView) {
        let (s1, s2) = self.split2();
        (s1.into_view(), s2.into_view())
    }

    /// Shuffles data points within [`DataSet`]
    pub fn shuffle(mut self) -> Self {
        self.points.shuffle(&mut GET_RNG());
        self
    }

    fn split3(mut self) -> (DataSet, DataSet, DataSet) {
        let mut training_set = DataSet::new(self.classes.clone());
        let mut verification_set = DataSet::new(self.classes.clone());
        let mut testing_set = DataSet::new(self.classes.clone());
        let test_points = self.points.split_off(self.points.len() * 2 / 3);
        let verification_points = self.points.split_off(self.points.len() / 2);
        let training_points = self.points;
        for point in training_points {
            training_set.add_data_point(point).unwrap();
        }
        for point in verification_points {
            verification_set.add_data_point(point).unwrap();
        }
        for point in test_points {
            testing_set.add_data_point(point).unwrap();
        }
        (training_set, verification_set, testing_set)
    }

    fn split2(mut self) -> (DataSet, DataSet) {
        let mut training_set = DataSet::new(self.classes.clone());
        let mut verification_set = DataSet::new(self.classes.clone());
        let verification_points = self.points.split_off(self.points.len() / 2);
        let training_points = self.points;
        for point in training_points {
            training_set.add_data_point(point).unwrap();
        }
        for point in verification_points {
            verification_set.add_data_point(point).unwrap();
        }
        (training_set, verification_set)
    }

    /// Reads data in Primeclue format from disk
    ///
    /// # Arguments
    /// * `path` - A [`Path`] pointing to a data directory inside which `data.ssd` file must exist.
    pub fn read_from_disk(path: &Path) -> Result<DataSet, PrimeclueErr> {
        let mut s = Serializator::load(&path.join(DATA_FILE_NAME))?;
        DataSet::deserialize(&mut s)
            .map_err(|e| PrimeclueErr::from(format!("Unable to deserialize data: {}", e)))
    }

    pub fn save_to_disk<F>(&self, path: &Path, mut callback: F) -> Result<(), PrimeclueErr>
    where
        F: FnMut(usize) -> Result<(), PrimeclueErr>,
    {
        if path.exists() {
            return PrimeclueErr::result(format!(
                "Directory {} already exists",
                path.to_str().unwrap()
            ));
        }
        fs::create_dir(path).map_err(|e| {
            format!("Unable to create directory {}, error: {}", path.to_str().unwrap(), e)
        })?;
        let mut serializator = Serializator::new();
        self.add_to_serializator(&mut callback, &mut serializator)?;
        serializator.save(&PathBuf::from(path), DATA_FILE_NAME)?;
        Ok(())
    }

    fn add_to_serializator<F>(
        &self,
        callback: &mut F,
        serializator: &mut Serializator,
    ) -> Result<(), PrimeclueErr>
    where
        F: FnMut(usize) -> Result<(), PrimeclueErr>,
    {
        serializator.add(&self.classes);
        serializator.add(&self.points.len());
        for (i, ds) in self.points.iter().enumerate() {
            serializator.add(ds);
            callback(i)?;
        }
        Ok(())
    }

    pub fn apply_rewards(&mut self, rewards: &Rewards) {
        self.points.iter_mut().for_each(|ds| {
            ds.outcome.set_reward_penalty(rewards.reward, rewards.penalty);
        });
    }
}

impl Serializable for DataSet {
    fn serialize(&self, s: &mut Serializator) {
        let mut callback = |_| Ok(());
        self.add_to_serializator(&mut callback, s).unwrap();
    }
}

impl Deserializable for DataSet {
    fn deserialize(s: &mut Serializator) -> Result<DataSet, String> {
        let classes = HashMap::deserialize(s)?;
        let len = usize::deserialize(s)?;
        let mut data = DataSet::new(classes);
        for _ in 0..len {
            let point = Point::deserialize(s)?;
            data.add_data_point(point)?;
        }
        Ok(data)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::data::data_set::{DataSet, Rewards};
    use crate::data::outcome::Class;
    use crate::data::{Input, Outcome, Point};
    use crate::rand::GET_RNG;
    use crate::serialization::serializator::test::test_serialization;
    use rand::Rng;
    use std::collections::HashMap;

    #[test]
    fn serialize() {
        let data = create_multiclass_data();
        test_serialization(data);
    }

    #[test]
    fn test_add_input_shape() {
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "false".to_string());
        classes.insert(Class::new(1), "true".to_string());
        let mut data = DataSet::new(classes);
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![1.0, 2.0]]).unwrap(),
            Outcome::new(Class::new(0), 1.0, -1.0),
        ))
        .unwrap();
        let result = data.add_data_point(Point::new(
            Input::from_vector(vec![vec![1.0, 2.0, 3.0]]).unwrap(),
            Outcome::new(Class::new(1), 1.0, -1.0),
        ));
        assert!(result.is_err())
    }

    #[test]
    fn test_empty() {
        let data = DataSet::new(HashMap::new());
        assert_eq!(data.is_empty(), true);
    }

    #[test]
    fn test_view() {
        let data = create_multiclass_data();
        let view = data.into_view();
        let expected: Vec<f32> = vec![1.0, 40.0, 10.0, 7.0, 100.0, 110.0, 13.0, 160.0, 60.0];
        assert_eq!(view.cells.get(0, 0), &expected);
        let expected: Vec<f32> = vec![2.0, 5.0, 20.0, 8.0, 11.0, 1.0, 14.0, 17.0, 7.0];
        assert_eq!(view.cells.get(0, 1), &expected);
        let expected: Vec<f32> = vec![3.0, 6.0, 30.0, 9.0, 12.0, 2.0, 15.0, 18.0, 8.0];
        assert_eq!(view.cells.get(0, 2), &expected);
        let expected: Vec<f32> = vec![4.0, 5.0, 11.0, 10.0, 11.0, 1.0, 10.0, 14.0, 4.0];
        assert_eq!(view.cells.get(1, 0), &expected);
        let expected: Vec<f32> = vec![5.0, 6.0, 12.0, 11.0, 12.0, 120.0, 11.0, 16.0, 6.0];
        assert_eq!(view.cells.get(1, 1), &expected);
        let expected: Vec<f32> = vec![6.0, 7.0, 13.0, 12.0, 13.0, 130.0, 12.0, 18.0, 8.0];
        assert_eq!(view.cells.get(1, 2), &expected);
    }

    #[test]
    fn test_split_with_marker() {
        let data = create_simple_data(1_000);
        let total = data.points.len();
        let marker = 500.0;
        let (t, v, tst) = data.split_with_test_data_marker(|p| p.input.get(0, 1) > marker);
        assert!(!t.outcomes.is_empty());
        assert!(!v.outcomes.is_empty());
        assert!(!tst.outcomes.is_empty());
        assert_eq!(t.outcomes.len() + v.outcomes.len() + tst.outcomes.len(), total);
        let above_marker = t.cells.get(0, 1).iter().any(|v| *v > marker);
        assert!(!above_marker);
        let above_marker = v.cells.get(0, 1).iter().any(|v| *v > marker);
        assert!(!above_marker);
        let below_marker = tst.cells.get(0, 1).iter().any(|v| *v <= marker);
        assert!(!below_marker);
    }

    #[test]
    fn test_multi_class_3splitting() {
        let data = create_big_multiclass_data();
        let exact_third = data.points.len() as f64 / 3.0;
        let per_set_per_class = exact_third / data.classes.len() as f64;
        let (tr, vs, tst) = data.shuffle().split3();
        assert_eq!(tr.points.len(), exact_third as usize);
        assert_eq!(tr.points.len(), tst.points.len());
        assert_eq!(tr.points.len(), vs.points.len());
        for class in 0..3 {
            let class = Class::new(class);
            let tr_count = tr.points.iter().filter(|p| p.outcome.class() == class).count();
            assert!(tr_count as f64 > 0.9 * per_set_per_class);
            let vs_count = vs.points.iter().filter(|p| p.outcome.class() == class).count();
            assert!(vs_count as f64 > 0.9 * per_set_per_class);
            let tst_count = tst.points.iter().filter(|p| p.outcome.class() == class).count();
            assert!(tst_count as f64 > 0.9 * per_set_per_class);
        }
    }

    #[test]
    fn test_multi_class_2splitting() {
        let data = create_big_multiclass_data();
        let exact_half = data.points.len() as f64 / 2.0;
        let per_set_per_class = exact_half / data.classes.len() as f64;
        let (ts, vs) = data.shuffle().split2();
        assert_eq!(ts.points.len(), exact_half as usize);
        assert_eq!(ts.points.len(), vs.points.len());
        for class in 0..3 {
            let class = Class::new(class);
            let tr_count = ts.points.iter().filter(|p| p.outcome.class() == class).count();
            assert!(tr_count as f64 > 0.9 * per_set_per_class);
            let vs_count = vs.points.iter().filter(|p| p.outcome.class() == class).count();
            assert!(vs_count as f64 > 0.9 * per_set_per_class);
        }
    }

    #[test]
    fn test_shuffling() {
        let data = create_simple_data(100);
        let (tr, _, tst) = data.shuffle().into_3_views_split();
        let tr_vec = tr.cells.get(0, 0);
        let tst_vec = tst.cells.get(0, 0);
        for (tr_value, tst_value) in tr_vec.iter().zip(tst_vec) {
            if tr_value > tst_value {
                return; // shuffled
            }
        }
        assert!(false, "Data not shuffled");
    }

    #[test]
    fn test_no_shuffle() {
        let data = create_simple_data(100);
        let (tr, _, tst) = data.into_3_views_split();
        let tr_vec = tr.cells.get(0, 0);
        let tst_vec = tst.cells.get(0, 0);
        for (tr_value, tst_value) in tr_vec.iter().zip(tst_vec) {
            assert!(tr_value < tst_value)
        }
    }

    #[test]
    fn rewards() {
        let mut data = create_multiclass_data();
        let rewards = Rewards { reward: 2.0, penalty: -3.0 };
        data.apply_rewards(&rewards);
        for point in &data.points {
            assert_eq!(point.outcome.reward(), 2.0);
            assert_eq!(point.outcome.penalty(), -3.0);
        }
        assert_eq!(data.cost_range(), (-27.0, 18.0))
    }

    #[test]
    fn filter() {
        let data = create_big_multiclass_data();
        let class = Class::new(0);
        let count = data.points.iter().filter(|p| p.outcome.class() == class).count();
        assert!(count > 0);
        let filtered = data.filter(|p| p.outcome.class() != class);
        let count = filtered.points.iter().filter(|p| p.outcome.class() == class).count();
        assert_eq!(count, 0);
    }

    pub(crate) fn create_simple_data(count: usize) -> DataSet {
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "FALSE".to_owned());
        classes.insert(Class::new(1), "TRUE".to_owned());
        let mut data = DataSet::new(classes);
        let mut rng = GET_RNG();
        for i in 0..count {
            let a = i as f32;
            let b = rng.gen_range(0.0, count as f32);
            data.add_data_point(Point::new(
                Input::from_vector(vec![vec![a, b]]).unwrap(),
                if a > b {
                    Outcome::new(Class::new(0), 1.0, -1.0)
                } else {
                    Outcome::new(Class::new(1), 1.0, -1.0)
                },
            ))
            .unwrap();
        }
        data
    }

    fn create_big_multiclass_data() -> DataSet {
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "a".to_owned());
        classes.insert(Class::new(1), "b".to_owned());
        classes.insert(Class::new(2), "c".to_owned());
        let mut data = DataSet::new(classes);
        let mut rng = GET_RNG();
        for _ in 0..3 * 3 * 10_000 {
            let a = rng.gen_range(0, 100);
            let b = rng.gen_range(0, 100);
            let c = rng.gen_range(0, 100);
            let class = if a > b && a > c {
                Class::new(0)
            } else if b > a && b > c {
                Class::new(1)
            } else {
                Class::new(2)
            };
            data.add_data_point(Point::new(
                Input::from_vector(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap(),
                Outcome::new(class, 1.0, -1.0),
            ))
            .unwrap();
        }
        data
    }

    pub(crate) fn create_multiclass_data() -> DataSet {
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "0".to_owned());
        classes.insert(Class::new(1), "1".to_owned());
        classes.insert(Class::new(2), "2".to_owned());
        let mut data = DataSet::new(classes);
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap(),
            Outcome::new(Class::new(0), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![40.0, 5.0, 6.0], vec![5.0, 6.0, 7.0]]).unwrap(),
            Outcome::new(Class::new(0), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![10.0, 20.0, 30.0], vec![11.0, 12.0, 13.0]]).unwrap(),
            Outcome::new(Class::new(0), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]]).unwrap(),
            Outcome::new(Class::new(1), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![100.0, 11.0, 12.0], vec![11.0, 12.0, 13.0]]).unwrap(),
            Outcome::new(Class::new(1), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![110.0, 1.0, 2.0], vec![1.0, 120.0, 130.0]]).unwrap(),
            Outcome::new(Class::new(1), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![13.0, 14.0, 15.0], vec![10.0, 11.0, 12.0]]).unwrap(),
            Outcome::new(Class::new(2), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![160.0, 17.0, 18.0], vec![14.0, 16.0, 18.0]]).unwrap(),
            Outcome::new(Class::new(2), 1.0, -1.0),
        ))
        .unwrap();
        data.add_data_point(Point::new(
            Input::from_vector(vec![vec![60.0, 7.0, 8.0], vec![4.0, 6.0, 8.0]]).unwrap(),
            Outcome::new(Class::new(2), 1.0, -1.0),
        ))
        .unwrap();
        data
    }
}
