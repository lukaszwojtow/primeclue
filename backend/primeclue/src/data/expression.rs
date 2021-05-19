// SPDX-License-Identifier: AGPL-3.0-or-later
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

use core::result::Result::{Err, Ok};
use std::cmp::Ordering;
use std::fmt::{Debug, Error, Formatter};

type Expression = Vec<String>;
type BoolProducer = Box<dyn Fn(f64, f64) -> bool>;
type Data<'a> = [Vec<&'a str>];
type ValueProducer = Box<dyn Fn(&Data<'_>, usize) -> Result<Option<f64>, String>>;
type PreProcessor = fn(&mut Expression, data: &Data<'_>) -> Result<(), String>;

pub fn parse(text: &str, data: &Data<'_>) -> Result<OutcomeProducer, String> {
    let input = text.to_owned();
    let mut expression: Expression = text
        .split(' ')
        .map(str::trim)
        .filter_map(|s| if s.is_empty() { None } else { Some(s.to_string()) })
        .collect();
    pre_process_expression(&mut expression, data)?;
    if expression.len() <= 4 {
        return Err(format!("Not enough tokens in expression: {:?}", expression));
    }
    let lhs = value_producer(&mut expression)?;
    let bool_producer = bool_producer(&mut expression)?;
    let rhs = value_producer(&mut expression)?;
    if !expression.is_empty() {
        return Err(format!("Not all text parsed {:?}", expression));
    }
    Ok(OutcomeProducer { input, lhs, rhs, bool_producer })
}

pub struct OutcomeProducer {
    input: String,
    lhs: ValueProducer,
    rhs: ValueProducer,
    bool_producer: BoolProducer,
}

impl Debug for OutcomeProducer {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.input)
    }
}

fn bool_producer(expression: &mut Expression) -> Result<BoolProducer, String> {
    match expression.remove(0).as_str() {
        "<=" => Ok(Box::new(|left, right| left <= right)),
        "<" => Ok(Box::new(|left, right| left < right)),
        "=" => Ok(Box::new(|left, right| (left - right).abs() < 0.001)),
        ">" => Ok(Box::new(|left, right| left > right)),
        ">=" => Ok(Box::new(|left, right| left >= right)),
        comparator => Err(format!("Invalid comparator in expression: {}", comparator)),
    }
}

fn first_as_f64(expression: &mut Expression) -> Result<f64, String> {
    check_empty_expression(expression)?;
    let first = expression.remove(0);
    parse_f64(&first)
}

fn parse_f64(s: &str) -> Result<f64, String> {
    let v =
        s.trim().parse::<f64>().map_err(|_| format!("Unable to parse '{}' to a number", s))?;
    if !v.is_finite() {
        Err(format!("Value '{}' is not finite", v))
    } else {
        Ok(v)
    }
}

fn first_as_index(expression: &mut Expression) -> Result<i32, String> {
    check_empty_expression(expression)?;
    let first = expression.remove(0);
    first.trim().parse().map_err(|_| format!("Unable to parse '{}' to an integer", first))
}

#[allow(clippy::ptr_arg)]
fn check_empty_expression(expression: &Expression) -> Result<(), String> {
    if expression.is_empty() {
        Err("Not enough tokens in expression".to_string())
    } else {
        Ok(())
    }
}

fn number_value_producer(expression: &mut Expression) -> Result<ValueProducer, String> {
    let value = first_as_f64(expression)?;
    Ok(Box::new(move |_, _| Ok(Some(value))))
}

fn column_value_producer(expression: &mut Expression) -> Result<ValueProducer, String> {
    let column = first_as_index(expression)?;
    column_check(column)?;
    Ok(Box::new(move |data, row| {
        let v = from_data(data, row, column as usize)?;
        Ok(Some(v))
    }))
}

fn cell_value_producer(expression: &mut Expression) -> Result<ValueProducer, String> {
    let row_offset = first_as_index(expression)?;
    let column = first_as_index(expression)?;
    column_check(column)?;
    Ok(Box::new(move |data, row| {
        let actual_row = row as i32 + row_offset;
        if actual_row < 0 || actual_row >= data.len() as i32 {
            Ok(None)
        } else {
            let v = from_data(data, actual_row as usize, column as usize)?;
            Ok(Some(v))
        }
    }))
}

fn column_check(column: i32) -> Result<(), String> {
    if column <= 0 {
        Err(format!("Column must be greater than 0, is: {}", column))
    } else {
        Ok(())
    }
}

fn from_data(data: &Data<'_>, row: usize, column: usize) -> Result<f64, String> {
    let actual_column = column - 1;
    if actual_column >= data[row].len() {
        Err(format!("Column {} does not exist", column))
    } else {
        parse_f64(data[row][actual_column])
    }
}

fn value_producer(expression: &mut Expression) -> Result<ValueProducer, String> {
    let name = expression.remove(0).to_uppercase();
    match name.as_str() {
        "CELL" => cell_value_producer(expression),
        "COLUMN" => column_value_producer(expression),
        "NUMBER" => number_value_producer(expression),
        _ => Err(format!("Invalid source type: {}", name)),
    }
}

impl OutcomeProducer {
    pub fn classify(&self, data: &Data<'_>, row: usize) -> Result<Option<bool>, String> {
        let left = (self.lhs)(data, row)?;
        let right = (self.rhs)(data, row)?;
        match (left, right) {
            (Some(left), Some(right)) => Ok(Some((self.bool_producer)(left, right))),
            _ => Ok(None),
        }
    }
}

// changes 'bool 5' into 'column 5 > number 0.5'
fn column_as_bool_preprocessor(expression: &mut Expression, _: &Data<'_>) -> Result<(), String> {
    if expression.len() == 2 && expression[0].to_uppercase() == "BOOL" {
        let column = expression[1].clone();
        column
            .parse::<usize>()
            .map_err(|_| "Bool column must be an integer greater than 0".to_string())?;
        expression.clear();
        expression.push("column".to_owned());
        expression.push(column);
        expression.push(">".to_owned());
        expression.push("number".to_owned());
        expression.push("0.5".to_owned());
    }
    Ok(())
}

// calculates average of column
// changes '... average 5... ' in expression into 'number <average of column 5>'
fn column_average_preprocessor(
    expression: &mut Expression,
    data: &Data<'_>,
) -> Result<(), String> {
    for (av_index, token) in expression.iter().enumerate() {
        if token.to_uppercase() == "AVERAGE" {
            let val_index = next_token_index(expression, av_index)?;
            let column = validate_column_index(&expression[val_index], data[0].len())?;
            let mut sum = 0.0;
            for i in 0..data.len() {
                sum += from_data(data, i, column)?;
            }
            let av = sum / data.len() as f64;
            expression[av_index] = "number".to_string();
            expression[val_index] = format!("{}", av);
            return Ok(());
        }
    }
    Ok(())
}

fn validate_column_index(txt: &str, length: usize) -> Result<usize, String> {
    let column =
        txt.parse::<usize>().map_err(|_| format!("Unable to parse '{}' to an integer", txt))?;
    if column > length || column == 0 {
        Err(format!("Invalid column: '{}'", column))
    } else {
        Ok(column)
    }
}

fn next_token_index(expression: &mut Vec<String>, index: usize) -> Result<usize, String> {
    let next_index = index + 1;
    if expression.len() <= next_index {
        Err("Not enough tokens in expression".to_string())
    } else {
        Ok(next_index)
    }
}

// calculates median of column
// changes '... median 5... ' in expression into 'number <median of column 5>'
fn column_median_preprocessor(
    expression: &mut Expression,
    data: &Data<'_>,
) -> Result<(), String> {
    for (token_index, token) in expression.iter().enumerate() {
        if token.to_uppercase() == "MEDIAN" {
            if data.len() < 2 {
                return Err("Not enough data to calculate median".to_string());
            }
            let val_index = next_token_index(expression, token_index)?;
            let column = validate_column_index(&expression[val_index], data[0].len())?;
            let mut values = Vec::with_capacity(data.len());
            for i in 0..data.len() {
                values.push(from_data(data, i, column)?);
            }
            values.sort_unstable_by(|v1, v2| v1.partial_cmp(v2).unwrap_or(Ordering::Equal));
            let med_index = values.len() / 2;
            let median = if values.len() % 2 == 0 {
                (values[med_index] + values[med_index - 1]) / 2.0
            } else {
                values[med_index]
            };
            expression[token_index] = "number".to_string();
            expression[val_index] = format!("{}", median);
            return Ok(());
        }
    }
    Ok(())
}

fn pre_process_expression(expression: &mut Expression, data: &Data<'_>) -> Result<(), String> {
    let pre_processors: Vec<PreProcessor> = vec![
        column_as_bool_preprocessor,
        column_average_preprocessor,
        column_median_preprocessor,
    ];
    for pp in pre_processors {
        pp(expression, data)?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::data::expression::parse;

    #[test]
    fn check_cell_row_offset() {
        let expression = "cell 0 1 < column 2";
        let data = vec![vec!["10.0", "5.0"], vec!["15.0", "25.0"], vec!["14.0", "10.0"]];
        let producer = parse(expression, &data).unwrap();
        let result = producer.classify(&data, 1).unwrap().unwrap();
        assert_eq!(result, true);

        let expression = "cell 0 1 = number 30";
        let data = vec![vec!["10.0", "5.0"], vec!["30.0", "25.0"], vec!["20.0", "15.0"]];
        let producer = parse(expression, &data).unwrap();
        let result = producer.classify(&data, 1).unwrap().unwrap();
        assert_eq!(result, true);
    }

    #[test]
    fn min_expression_len() {
        assert_expression_error("1 2 3 4");
    }

    #[test]
    fn invalid_comparator() {
        assert_expression_error("column 1 != number 4");
    }

    #[test]
    fn non_existent_column() {
        let e = "column 2 > number 3";
        let data = vec![vec!["3.0"]];
        let op = parse(e, &data).unwrap();
        let c = op.classify(&data, 0);
        assert_eq!(c.is_err(), true);
        assert_expression_error("column 0 > number 3");
        assert_expression_error("column -1 > number 3");
    }

    #[test]
    fn trims_input() {
        let e = "     column   1 <  number    5   ";
        let data = vec![vec!["4.0", "10.0"]];
        let op = parse(e, &data).unwrap();
        let b = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(b, true);
    }

    #[test]
    fn no_outcome_for_non_existent_row() {
        let e = "cell 1 1 > number 3";
        let data = vec![vec!["1.0"]];
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap();
        assert_eq!(r.is_none(), true);

        let e = "cell -1 1 > number 3";
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap();
        assert_eq!(r.is_none(), true);
    }

    #[test]
    fn error_for_empty_expression() {
        assert_expression_error("");
    }

    #[test]
    fn error_on_too_long_expression() {
        assert_expression_error("column 1 > number 2 42");
    }

    #[test]
    fn check_bool_preprocessor() {
        let e = "bool 2";
        let data = vec![vec!["5.0", "-1.5"], vec!["4.0", "0.0"], vec!["6.0", "1.0"]];
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, true);
    }

    #[test]
    fn check_average_preprocessor() {
        let e = "column 1 < average";
        assert_expression_error(e);
        let data = vec![vec!["5.0", "10.0"], vec!["4.0", "10.0"], vec!["6.0", "10.0"]];
        let e = "column 1 < average 3";
        assert_eq!(parse(e, &data).is_err(), true);
    }

    #[test]
    fn check_nan_inf() {
        let data = vec![vec!["inf", "10.0"], vec!["-inf", "10.0"], vec!["nan", "10.0"]];
        let e = "column 1 > number 0";
        let op = parse(e, &data).unwrap();
        assert!(op.classify(&data, 0).is_err());
        assert!(op.classify(&data, 1).is_err());
        assert!(op.classify(&data, 2).is_err());
    }

    #[test]
    fn check_median_preprocessor() {
        let e = "column 1 < median";
        assert_expression_error(e);

        let data = vec![vec!["5.0", "10.0"]];
        let e = "column 1 < median 1";
        assert_eq!(parse(e, &data).is_err(), true);

        let data = vec![vec!["1.0", "10.0"], vec!["4.0", "10.0"], vec!["6.0", "10.0"]];
        let e = "column 1 < median 1";
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, false);

        let data = vec![
            vec!["1.0", "4.0"],
            vec!["4.0", "4.5"],
            vec!["6.0", "5.5"],
            vec!["8.0", "10.0"],
        ];
        let e = "column 2 < median 1";
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 3).unwrap().unwrap();
        assert_eq!(r, false);
    }

    #[test]
    fn check_less_than() {
        let e = "column 1 < number 5";
        let data = vec![vec!["5.0", "10.0"], vec!["4.0", "10.0"], vec!["6.0", "10.0"]];
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, false);
    }

    #[test]
    fn check_eq() {
        let e = "column 1 = number 5";
        let data = vec![vec!["5.0", "10.0"], vec!["4.0", "10.0"], vec!["6.0", "10.0"]];
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, false);
    }

    #[test]
    fn check_greater_than() {
        let e = "column 1 > number 5";
        let data = vec![vec!["5.0", "10.0"], vec!["4.0", "10.0"], vec!["6.0", "10.0"]];
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, false);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, true);
    }

    #[test]
    fn check_less_than_equal() {
        let e = "column 1 <= number 5";
        let data = vec![vec!["5.0", "10.0"], vec!["4.0", "10.0"], vec!["6.0", "10.0"]];
        let op = parse(e, &data).unwrap();
        let r = op.classify(&data, 0).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 1).unwrap().unwrap();
        assert_eq!(r, true);

        let r = op.classify(&data, 2).unwrap().unwrap();
        assert_eq!(r, false);
    }

    #[test]
    fn check_greater_than_equal() {
        let expression = "column 1 >= number 5";
        let data = vec![vec!["10.0", "1.0"]];
        let producer = parse(expression, &data).unwrap();
        let result = producer.classify(&data, 0).unwrap().unwrap();
        assert_eq!(result, true);

        let data = vec![vec!["1.0", "10.0"]];
        let result = producer.classify(&data, 0).unwrap().unwrap();
        assert_eq!(result, false);
    }

    #[test]
    fn check_average_of_column() {
        let expression = "column 1 > average 2";
        let data =
            vec![vec!["1.0", "1.0"], vec!["2.0", "2.0"], vec!["3.0", "3.0"], vec!["4.0", "4.0"]];
        let producer = parse(expression, &data).unwrap();
        let result = producer.classify(&data, 0).unwrap().unwrap();
        assert_eq!(result, false);
        let result = producer.classify(&data, 1).unwrap().unwrap();
        assert_eq!(result, false);
        let result = producer.classify(&data, 2).unwrap().unwrap();
        assert_eq!(result, true);
        let result = producer.classify(&data, 3).unwrap().unwrap();
        assert_eq!(result, true);
    }

    #[test]
    fn check_nan_in_data() {
        let nan = "NaN";
        let n: f64 = nan.parse().unwrap();
        assert!(n.is_nan(), "{}", true);
        let expression = "column 1 < column 2";
        let data = vec![vec![nan, "2.0"]];
        let producer = parse(expression, &data).unwrap();
        let result = producer.classify(&data, 0);
        assert!(result.is_err())
    }

    #[test]
    fn index_not_integer() {
        assert_expression_error("column 0.1 > number 1");
        assert_expression_error("cell 0.1 1 > number 1");
        assert_expression_error("cell 1 0.1 > number 1");
    }

    #[test]
    fn not_enough_expression_for_number() {
        assert_expression_error("cell 0 1 < number");
    }

    #[test]
    fn not_enough_expression_for_cell() {
        assert_expression_error("cell 0 1 < cell 0");
    }

    #[test]
    fn not_enough_expression_for_column() {
        assert_expression_error("cell 0 1 < column");
    }

    #[test]
    fn negative_column() {
        assert_expression_error("cell 0 1 < column -1");
    }

    #[test]
    fn non_float_number() {
        assert_expression_error("cell 0 1 < number NotAFloat");
    }

    #[test]
    fn no_average_column() {
        assert_expression_error("column 1 > average")
    }

    #[test]
    fn non_float_average_column() {
        assert_expression_error("column 1 > average NotAFloat")
    }

    #[test]
    fn invalid_source() {
        assert_expression_error("cell 0 1 < tomorrow 1");
    }

    fn assert_expression_error(e: &str) {
        let data = vec![vec!["0.0"]];
        assert_eq!(parse(e, &data).is_err(), true);
    }
}
