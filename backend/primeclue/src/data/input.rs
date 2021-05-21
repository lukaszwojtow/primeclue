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

use crate::data::Data;
use crate::data::InputShape;
use crate::error::PrimeclueErr;
use crate::serialization::{Deserializable, Serializable, Serializator};

#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    data: Data<f32>,
}
impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

impl Input {
    #[must_use]
    pub fn new() -> Input {
        Input { data: Data::new() }
    }

    pub fn from_vector(data: Vec<Vec<f32>>) -> Result<Input, PrimeclueErr> {
        let mut input_data = Input::new();

        for (id, row) in data.into_iter().enumerate() {
            input_data.add_row(row).map_err(|e| {
                PrimeclueErr::from(format!("Unable to create input row {}: {} ", id + 1, e))
            })?;
        }
        Ok(input_data)
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> f32 {
        *self.data.get(row, column)
    }

    pub fn add_row(&mut self, row: Vec<f32>) -> Result<usize, PrimeclueErr> {
        self.data.add_row(row)
    }

    #[must_use]
    pub fn input_shape(&self) -> &InputShape {
        self.data.input_shape()
    }

    #[must_use]
    pub fn row(&self, r: usize) -> Vec<f32> {
        self.data.row(r).into_iter().copied().collect()
    }

    #[must_use]
    pub fn to_view(&self) -> Data<Vec<f32>> {
        let mut data = Data::new();
        for row in 0..self.input_shape().rows() {
            let mut vec = vec![];
            for col in 0..self.input_shape().columns() {
                vec.push(vec![self.get(row, col)])
            }
            data.add_row(vec).unwrap();
        }
        data
    }
}

impl Serializable for Input {
    fn serialize(&self, s: &mut Serializator) {
        s.add(self.data.input_shape());
        for i in 0..self.input_shape().rows() {
            let row = self.row(i);
            row.serialize(s);
        }
    }
}

impl Deserializable for Input {
    fn deserialize(s: &mut Serializator) -> Result<Input, String> {
        let mut id = Input::new();
        let input_shape = InputShape::deserialize(s)?;
        for _ in 0..input_shape.rows() {
            let row = Vec::deserialize(s)?;
            id.add_row(row).map_err(|e| e.to_string())?;
        }
        Ok(id)
    }
}

#[cfg(test)]
mod test {
    use crate::data::Input;

    #[test]
    fn test_input_to_view_single_row() {
        let vec = vec![vec![1.0, 2.0, 3.0]];
        let input = Input::from_vector(vec).unwrap();
        let view = input.to_view();
        assert_eq!(view.get(0, 0), &vec![1.0]);
        assert_eq!(view.get(0, 1), &vec![2.0]);
        assert_eq!(view.get(0, 2), &vec![3.0]);
    }

    #[test]
    fn test_input_to_view_many_rows() {
        let vec = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let input = Input::from_vector(vec).unwrap();
        let view = input.to_view();
        assert_eq!(view.get(0, 0), &vec![1.0]);
        assert_eq!(view.get(0, 1), &vec![2.0]);
        assert_eq!(view.get(0, 2), &vec![3.0]);

        assert_eq!(view.get(1, 0), &vec![4.0]);
        assert_eq!(view.get(1, 1), &vec![5.0]);
        assert_eq!(view.get(1, 2), &vec![6.0]);

        assert_eq!(view.get(2, 0), &vec![7.0]);
        assert_eq!(view.get(2, 1), &vec![8.0]);
        assert_eq!(view.get(2, 2), &vec![9.0]);
    }
}
