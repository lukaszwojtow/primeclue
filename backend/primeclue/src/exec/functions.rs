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

use crate::serialization::{Deserializable, Serializable, Serializator};
use std::cmp::Ordering;
use std::f32::consts::*;
use std::fmt::{Debug, Error, Formatter};

pub static MATH_CONSTANTS: [MathConst; 6] = [
    MathConst { name: "0", value: 0.0 },
    MathConst { name: "1", value: 1.0 },
    MathConst { name: "2", value: 2.0 },
    MathConst { name: "e", value: E },
    MathConst { name: "pi", value: PI },
    MathConst { name: "2pi", value: 2.0 * PI },
];

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct MathConst {
    name: &'static str,
    value: f32,
}

impl MathConst {
    pub fn value(&self) -> f32 {
        self.value
    }
}

impl Serializable for MathConst {
    fn serialize(&self, s: &mut Serializator) {
        s.add_str(self.name)
    }
}

impl Deserializable for &MathConst {
    fn deserialize(s: &mut Serializator) -> Result<&'static MathConst, String> {
        let v = s.next_token()?;
        for c in &MATH_CONSTANTS {
            if c.name.eq(v) {
                return Ok(c);
            }
        }
        Err(format!("MathConst {} not found", v))
    }
}

fn single_array_fun(f: fn(f32) -> f32, mut v: Vec<f32>) -> Vec<f32> {
    for value in &mut v {
        *value = f(*value);
    }
    v
}

fn sqrt_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(sqrt, v)
}

fn sqrt(v: f32) -> f32 {
    v.sqrt()
}

fn square_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(square, v)
}

fn square(v: f32) -> f32 {
    v * v
}

fn log_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(log, v)
}

fn log(v: f32) -> f32 {
    v.log(E)
}

fn reciprocal_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(reciprocal, v)
}

fn reciprocal(v: f32) -> f32 {
    1.0 / v
}

fn sine_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(sine, v)
}

fn sine(v: f32) -> f32 {
    f32::sin(v)
}

fn abs_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(abs, v)
}

fn ceil_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(ceil, v)
}

fn ceil(v: f32) -> f32 {
    v.ceil()
}

fn abs(v: f32) -> f32 {
    v.abs()
}

fn inc_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(inc, v)
}

fn inc(v: f32) -> f32 {
    v + 1.0
}

fn dec_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(dec, v)
}

fn floor_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(floor, v)
}

fn dec(v: f32) -> f32 {
    v - 1.0
}

fn floor(v: f32) -> f32 {
    v.floor()
}

fn neg_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(neg, v)
}

fn neg(v: f32) -> f32 {
    -v
}

fn to_one_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(to_one, v)
}

fn to_one(v: f32) -> f32 {
    match v.partial_cmp(&0.0) {
        Some(Ordering::Equal) | None => 0.0,
        Some(Ordering::Greater) => 1.0,
        Some(Ordering::Less) => -1.0,
    }
}

fn tau_sigmoid_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(tau_sigmoid, v)
}

fn tau_sigmoid(v: f32) -> f32 {
    (2.0 * PI).powf(v)
}

fn tang_hyper_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(tang_hyper, v)
}

fn tang_hyper(v: f32) -> f32 {
    let e_pos = E.powf(v);
    let e_neg = E.powf(-v);
    (e_pos - e_neg) / (e_pos + e_neg)
}

fn relu_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(relu, v)
}

fn round_array(v: Vec<f32>) -> Vec<f32> {
    single_array_fun(round, v)
}

fn round(v: f32) -> f32 {
    v.round()
}

fn relu(v: f32) -> f32 {
    v.max(0.0)
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct SingleArgFunction {
    pub name: &'static str,
    pub fun: fn(Vec<f32>) -> Vec<f32>,
}

impl Serializable for SingleArgFunction {
    fn serialize(&self, s: &mut Serializator) {
        s.add_str(self.name);
    }
}

impl Deserializable for &SingleArgFunction {
    fn deserialize(s: &mut Serializator) -> Result<&'static SingleArgFunction, String> {
        let v = s.next_token()?;
        for fun in &ONE_ARG_FUNCTIONS {
            if fun.name.eq(v) {
                return Ok(fun);
            }
        }
        Err(format!("SingleArgFunction {} not found", v))
    }
}

pub static ONE_ARG_FUNCTIONS: [SingleArgFunction; 16] = [
    SingleArgFunction { name: "abs", fun: abs_array },
    SingleArgFunction { name: "ceil", fun: ceil_array },
    SingleArgFunction { name: "dec", fun: dec_array },
    SingleArgFunction { name: "floor", fun: floor_array },
    SingleArgFunction { name: "inc", fun: inc_array },
    SingleArgFunction { name: "log", fun: log_array },
    SingleArgFunction { name: "neg", fun: neg_array },
    SingleArgFunction { name: "normalize", fun: to_one_array },
    SingleArgFunction { name: "reciprocal", fun: reciprocal_array },
    SingleArgFunction { name: "relu", fun: relu_array },
    SingleArgFunction { name: "round", fun: round_array },
    SingleArgFunction { name: "sine", fun: sine_array },
    SingleArgFunction { name: "sqrt", fun: sqrt_array },
    SingleArgFunction { name: "square", fun: square_array },
    SingleArgFunction { name: "tau_sigmoid", fun: tau_sigmoid_array },
    SingleArgFunction { name: "tang_hyper", fun: tang_hyper_array },
];

fn two_arrays_fun(f: fn(f32, f32) -> f32, mut v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    for (value1, value2) in v1.iter_mut().zip(v2) {
        *value1 = f(*value1, *value2);
    }
    v1
}

fn add_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(add, v1, v2)
}
fn add(v1: f32, v2: f32) -> f32 {
    v1 + v2
}

fn sub_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(sub, v1, v2)
}
fn sub(v1: f32, v2: f32) -> f32 {
    v1 - v2
}

fn div_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(div, v1, v2)
}
fn div(v1: f32, v2: f32) -> f32 {
    v1 / v2
}

fn mul_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(mul, v1, v2)
}
fn mul(v1: f32, v2: f32) -> f32 {
    v1 * v2
}

fn higher_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(higher, v1, v2)
}
fn higher(v1: f32, v2: f32) -> f32 {
    v1.max(v2)
}

fn lower_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(lower, v1, v2)
}
fn lower(v1: f32, v2: f32) -> f32 {
    v1.min(v2)
}

fn equal_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(equal, v1, v2)
}

fn equal(v1: f32, v2: f32) -> f32 {
    if (1.0 - v1 / v2).abs() < 0.01 {
        1.0
    } else {
        0.0
    }
}

fn abs_higher_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(abs_higher, v1, v2)
}
fn abs_higher(v1: f32, v2: f32) -> f32 {
    v1.abs().max(v2.abs())
}

fn abs_lower_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(abs_lower, v1, v2)
}
fn abs_lower(v1: f32, v2: f32) -> f32 {
    v1.abs().min(v2.abs())
}

fn mid_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(mid, v1, v2)
}
fn mid(v1: f32, v2: f32) -> f32 {
    (v1 + v2) / 2.0
}

fn sum_of_squares_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(sum_of_squares, v1, v2)
}
fn sum_of_squares(v1: f32, v2: f32) -> f32 {
    v1 * v1 + v2 * v2
}

fn first_is_higher_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(first_is_higher, v1, v2)
}
fn first_is_higher(v1: f32, v2: f32) -> f32 {
    if v1 > v2 {
        1.0
    } else {
        0.0
    }
}

fn xor_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(xor, v1, v2)
}
fn xor(v1: f32, v2: f32) -> f32 {
    if (v1 == 0.0 && v2 != 0.0) || (v1 != 0.0 && v2 == 0.0) {
        1.0
    } else {
        0.0
    }
}

fn or_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(or, v1, v2)
}
fn or(v1: f32, v2: f32) -> f32 {
    if v1 != 0.0 || v2 != 0.0 {
        1.0
    } else {
        0.0
    }
}

fn round_equal_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(round_equal, v1, v2)
}

#[allow(clippy::float_cmp)]
fn round_equal(v1: f32, v2: f32) -> f32 {
    if v1.round() == v2.round() {
        1.0
    } else {
        0.0
    }
}

fn and_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(and, v1, v2)
}
fn and(v1: f32, v2: f32) -> f32 {
    if v1 != 0.0 && v2 != 0.0 {
        1.0
    } else {
        0.0
    }
}

fn diff_array(v1: Vec<f32>, v2: &[f32]) -> Vec<f32> {
    two_arrays_fun(diff, v1, v2)
}
fn diff(v1: f32, v2: f32) -> f32 {
    (1.0 - v1 / v2).abs()
}

#[derive(Copy, Clone)]
pub struct DoubleArgFunction {
    pub name: &'static str,
    pub fun: fn(Vec<f32>, &[f32]) -> Vec<f32>,
}

impl Debug for DoubleArgFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.name)
    }
}

impl PartialEq for DoubleArgFunction {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Serializable for DoubleArgFunction {
    fn serialize(&self, s: &mut Serializator) {
        s.add_str(self.name);
    }
}

impl Deserializable for &DoubleArgFunction {
    fn deserialize(s: &mut Serializator) -> Result<&'static DoubleArgFunction, String> {
        let v = s.next_token()?;
        for fun in &TWO_ARG_FUNCTIONS {
            if fun.name.eq(v) {
                return Ok(fun);
            }
        }
        Err(format!("DoubleArgFunction {} not found", v))
    }
}

pub static TWO_ARG_FUNCTIONS: [DoubleArgFunction; 17] = [
    DoubleArgFunction { name: "abs_higher", fun: abs_higher_array },
    DoubleArgFunction { name: "abs_lower", fun: abs_lower_array },
    DoubleArgFunction { name: "add", fun: add_array },
    DoubleArgFunction { name: "and", fun: and_array },
    DoubleArgFunction { name: "diff", fun: diff_array },
    DoubleArgFunction { name: "div", fun: div_array },
    DoubleArgFunction { name: "equal", fun: equal_array },
    DoubleArgFunction { name: "first_is_higher", fun: first_is_higher_array },
    DoubleArgFunction { name: "higher", fun: higher_array },
    DoubleArgFunction { name: "lower", fun: lower_array },
    DoubleArgFunction { name: "mid", fun: mid_array },
    DoubleArgFunction { name: "mul", fun: mul_array },
    DoubleArgFunction { name: "or", fun: or_array },
    DoubleArgFunction { name: "sub", fun: sub_array },
    DoubleArgFunction { name: "sum_of_squares", fun: sum_of_squares_array },
    DoubleArgFunction { name: "xor", fun: xor_array },
    DoubleArgFunction { name: "round_equal_array", fun: round_equal_array },
];

#[cfg(test)]
mod test {
    use crate::exec::functions::{equal, relu};

    #[test]
    fn test_equal() {
        let v1 = 1.0;
        let v2 = v1;
        assert!((1.0 - equal(v1, v2)).abs() < f32::EPSILON);

        let v1 = -1.0;
        let v2 = 1.0;
        assert!((0.0 - equal(v1, v2)).abs() < f32::EPSILON);

        let v1 = 1010.0;
        let v2 = 1000.0;
        assert!((1.0 - equal(v1, v2)).abs() < f32::EPSILON);

        let v1 = 100.0;
        let v2 = 1000.0;
        assert!((0.0 - equal(v1, v2)).abs() < f32::EPSILON);

        let v1 = 1020.0;
        let v2 = 1000.0;
        assert!((0.0 - equal(v1, v2)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_relu() {
        assert!((relu(1.0) - 1.0).abs() < f32::EPSILON);
        assert!((relu(-15.0) - 0.0).abs() < f32::EPSILON);
        assert!((relu(15.0) - 15.0).abs() < f32::EPSILON);
    }
}
