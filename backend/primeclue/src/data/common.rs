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

use crate::contrand::GET_RNG;
use crate::error::PrimeclueErr;
use crate::serialization::{Deserializable, Serializable, Serializator};
use rand::Rng;
use std::fmt::{Debug, Error, Formatter};

#[derive(Clone)]
pub struct Data<T> {
    input_shape: InputShape,
    data: Vec<T>,
}

impl<T: Debug> Debug for Data<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Shape: {}, Data: {:?}", self.input_shape, self.data)
    }
}

impl<T: PartialEq> PartialEq for Data<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.input_shape == other.input_shape
    }
}

impl<T> Default for Data<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Data<T> {
    #[must_use]
    pub fn new() -> Data<T> {
        Data { input_shape: InputShape::new(0, 0), data: Vec::new() }
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> &T {
        &self.data[row * self.input_shape.columns + column]
    }

    pub fn add_row(&mut self, row: Vec<T>) -> Result<usize, PrimeclueErr> {
        if !self.data.is_empty() && self.input_shape.columns() != row.len() {
            PrimeclueErr::result(format!(
                "Invalid number of columns, found: {}, required: {}",
                row.len(),
                self.input_shape.columns()
            ))
        } else {
            self.input_shape.update(self.input_shape.rows() + 1, row.len());
            self.data.extend(row.into_iter());
            Ok(self.input_shape.rows())
        }
    }

    #[must_use]
    pub fn input_shape(&self) -> &InputShape {
        &self.input_shape
    }

    #[must_use]
    pub fn row(&self, r: usize) -> Vec<&T> {
        let start = r * self.input_shape.columns();
        let end = start + self.input_shape.columns();
        self.data[start..end].iter().collect()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputShape {
    rows: usize,
    columns: usize,
}

impl std::fmt::Display for InputShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "rows: {}, columns: {}", self.rows, self.columns)
    }
}

impl InputShape {
    #[must_use]
    pub fn new(rows: usize, columns: usize) -> InputShape {
        InputShape { rows, columns }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn update(&mut self, rows: usize, columns: usize) {
        self.rows = rows;
        self.columns = columns;
    }

    pub fn random_row_column(&self, forbidden_cols: &[usize]) -> (usize, usize) {
        let mut rng = GET_RNG();
        loop {
            let col = rng.gen_range(0..self.columns);
            if !forbidden_cols.contains(&col) {
                return (rng.gen_range(0..self.rows), col);
            }
        }
    }
}

impl Serializable for InputShape {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.rows, &self.columns]);
    }
}

impl Deserializable for InputShape {
    fn deserialize(s: &mut Serializator) -> Result<InputShape, String> {
        let next = s.next_token()?;
        let rows = next.parse().map_err(|e| format!("Unable to parse {}: {}", next, e))?;
        let next = s.next_token()?;
        let columns = next.parse().map_err(|e| format!("Unable to parse {}: {}", next, e))?;
        Ok(InputShape::new(rows, columns))
    }
}

#[cfg(test)]
mod test {
    use crate::contrand::GET_RNG;
    use crate::data::{Data, InputShape};
    use rand::Rng;

    #[test]
    fn add_row() {
        let mut d = Data::new();
        d.add_row(vec![1.0, 2.0]).unwrap();
        let r = d.add_row(vec![1.0, 2.0, 3.0]);
        assert!(r.is_err());
        let r = d.add_row(vec![1.0]);
        assert!(r.is_err())
    }

    #[test]
    fn random_with_forbidden() {
        let mut rng = GET_RNG();
        let columns = 100;
        let shape = InputShape::new(1, columns);
        for _ in 0..100 {
            let mut forbidden = vec![];
            for _ in 1..rng.gen_range(2..columns) {
                forbidden.push(rng.gen_range(0..columns))
            }
            for _ in 0..100 {
                let (_, column) = shape.random_row_column(&forbidden);
                assert!(!forbidden.contains(&column))
            }
        }
    }
}
