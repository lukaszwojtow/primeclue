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

use crate::data::data_set::DataSet;
use crate::data::outcome::Class;
use crate::data::{Input, Outcome, Point};
use crate::error::PrimeclueErr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn create_input_data(
    line: usize,
    floats: &[Vec<f32>],
    rows_per_set: usize,
) -> Result<Input, PrimeclueErr> {
    let start_line = 1 + line - rows_per_set;

    let mut id = Input::new();
    for floats_line in floats[start_line..=line].iter() {
        id.add_row(floats_line.clone()).map_err(|e| {
            PrimeclueErr::from(format!("Unable to import line: {:?}: {}", floats_line, e))
        })?;
    }
    Ok(id)
}

#[derive(Deserialize, Debug)]
pub struct ClassRequest {
    // TODO remove pub
    pub content: String,
    pub class_column: usize,
    pub separator: String,
    pub ignore_first_row: bool,
    pub rows_per_set: usize,
    pub import_columns: Vec<bool>,
    pub data_name: String,
    pub custom_reward_penalty_columns: bool,
    pub reward_column: usize,
    pub penalty_column: usize,
}

impl ClassRequest {
    fn extract_reward_penalty(&self, row: &[&str]) -> Result<(f32, f32), PrimeclueErr> {
        if self.custom_reward_penalty_columns {
            let reward = row.get(self.reward_column - 1).ok_or_else(|| {
                PrimeclueErr::from(format!("Unable to access {}'nth column", self.reward_column))
            })?;
            let reward = reward.trim().parse().map_err(|_| {
                PrimeclueErr::from(format!("Unable to parse '{}' to reward", reward))
            })?;
            let penalty = row.get(self.penalty_column - 1).ok_or_else(|| {
                PrimeclueErr::from(format!(
                    "Unable to access {}'nth column",
                    self.penalty_column
                ))
            })?;
            let penalty = penalty.trim().parse().map_err(|_| {
                PrimeclueErr::from(format!("Unable to parse '{}' to penalty", penalty))
            })?;
            Ok((reward, penalty))
        } else {
            Ok((1.0, -1.0))
        }
    }

    /// Method to build [`ClassRequest`] for a simple case:
    /// - use comma (,) as field separator
    /// - import all but last column
    /// - use the last column as class label
    ///
    /// # Arguments:
    /// * `name`: name of imported data
    /// * `content`: data string: '1.0,3.0,false\r\n2.0,1.0,true\r\n'
    /// * `ignore_first_row`: use first row in content as header (do not import)
    pub fn simple_csv_request(name: &str, content: String, ignore_first_row: bool) -> Self {
        let line = &split_to_vec(&content, ",", ignore_first_row)[0];
        let len = line.len();
        let mut import_columns = vec![true; len];
        let last = import_columns.get_mut(len - 1).unwrap();
        *last = false;

        ClassRequest {
            content,
            class_column: len,
            separator: ",".to_string(),
            ignore_first_row,
            rows_per_set: 1,
            import_columns,
            data_name: name.to_string(),
            custom_reward_penalty_columns: false,
            reward_column: 0,
            penalty_column: 0,
        }
    }
}

pub fn build_data_set(r: &ClassRequest) -> Result<DataSet, PrimeclueErr> {
    let data = split_to_vec(&r.content, &r.separator, r.ignore_first_row);
    let class_producer = class_producer(r, &data)?;
    let mut numbers = vec![];
    let mut data_set = DataSet::new(class_producer.all_classes());
    for (row_num, row) in data.iter().enumerate() {
        numbers.push(build_numbers_row(&r.import_columns, row_num, row)?);
        if row_num + 1 < r.rows_per_set {
            continue;
        }
        if let Some(outcome) = class_producer.class(&data, row_num)? {
            let (reward, penalty) = r.extract_reward_penalty(row)?;
            data_set.add_data_point(build_data_point(
                r,
                &mut numbers,
                row_num,
                outcome,
                reward,
                penalty,
            )?)?;
        }
    }
    Ok(data_set)
}

fn build_data_point(
    r: &ClassRequest,
    numbers: &mut [Vec<f32>],
    row_num: usize,
    outcome: Class,
    reward: f32,
    penalty: f32,
) -> Result<Point, PrimeclueErr> {
    let id = create_input_data(row_num, numbers, r.rows_per_set)?;
    let pd = Outcome::new(outcome, reward, penalty);
    Ok(Point::new(id, pd))
}

pub fn get_header_row(
    content: &str,
    separator: &str,
    ignore_first_row: bool,
    mut names: Vec<String>,
) -> Vec<String> {
    let lines = content.split('\n').map(str::trim).collect::<Vec<_>>();
    match lines.first() {
        None => vec![],
        Some(first) => {
            let mut header = if ignore_first_row {
                first.split(separator).map(|s| s.to_owned()).collect::<Vec<_>>()
            } else {
                vec!["".to_owned(); first.split(separator).count()]
            };
            header.append(&mut names);
            header
        }
    }
}
pub fn remove_column<'a>(
    content: &mut [Vec<&'a str>],
    column: usize,
) -> Result<Vec<&'a str>, PrimeclueErr> {
    let mut col_values = Vec::with_capacity(content.len());
    for row in content.iter_mut() {
        if column >= row.len() {
            return PrimeclueErr::result(format!(
                "Unable to remove {}th column, only {} columns present",
                column,
                row.len()
            ));
        }
        let v = row.remove(column);
        col_values.push(v);
    }
    Ok(col_values)
}

pub fn split_to_vec<'a>(
    content: &'a str,
    separator: &str,
    ignore_first_row: bool,
) -> Vec<Vec<&'a str>> {
    let mut rows: Vec<&str> =
        content.split('\n').map(str::trim).filter(|s| !s.is_empty()).collect();
    if ignore_first_row && !rows.is_empty() {
        rows.remove(0);
    }
    rows.iter().map(|r| r.split(separator).collect()).collect()
}

#[derive(Debug)]
pub struct ClassProducer {
    column: usize,
    classes: HashMap<String, Class>,
}

impl ClassProducer {
    pub fn create(column: usize, classes: HashMap<String, Class>) -> Self {
        ClassProducer { column, classes }
    }

    pub fn class(&self, data: &[Vec<&str>], row: usize) -> Result<Option<Class>, PrimeclueErr> {
        let v = data[row][self.column];
        Ok(Some(*self.classes.get(v).unwrap()))
    }

    fn all_classes(&self) -> HashMap<Class, String> {
        let mut classes = HashMap::new();
        for (k, v) in &self.classes {
            classes.insert(*v, k.to_owned());
        }
        classes
    }
}

pub fn class_producer(
    r: &ClassRequest,
    data: &[Vec<&str>],
) -> Result<ClassProducer, PrimeclueErr> {
    let column = r.class_column - 1;
    let classes = build_class_map(data, column)?;
    Ok(ClassProducer::create(column, classes))
}

pub fn build_numbers_row(
    use_columns: &[bool],
    row_num: usize,
    row: &[&str],
) -> Result<Vec<f32>, PrimeclueErr> {
    let to_import = use_columns
        .iter()
        .zip(row)
        .filter_map(|(&keep, &s)| if keep { Some(s) } else { None })
        .collect::<Vec<&str>>();
    let mut num_row = vec![];
    for value in to_import {
        let n = value.trim().parse().map_err(|err| {
            format!("Unable to parse '{}' to number: row {}: {}", value, row_num + 1, err)
        })?;
        num_row.push(n);
    }
    Ok(num_row)
}

pub fn build_class_map(
    data: &[Vec<&str>],
    column: usize,
) -> Result<HashMap<String, Class>, PrimeclueErr> {
    let mut classes = HashMap::new();
    for (i, row) in data.iter().enumerate() {
        let v = row
            .get(column)
            .ok_or_else(|| PrimeclueErr::from(format!("No column {} in row {}", column, i)))?;
        if !v.is_empty() && !classes.contains_key(*v) {
            classes.insert(v.to_string(), Class::new(classes.len() as u16));
        }
    }
    Ok(classes)
}

#[derive(Serialize, Debug)]
pub struct ClassResponse {
    classes: Vec<String>,
}

impl ClassResponse {
    pub fn new(classes: Vec<String>) -> Self {
        ClassResponse { classes }
    }
}
