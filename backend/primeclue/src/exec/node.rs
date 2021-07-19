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
use crate::data::{Data, InputShape};
use crate::exec::functions::{DoubleArgFunction, MathConst, SingleArgFunction};
use crate::exec::functions::{MATH_CONSTANTS, ONE_ARG_FUNCTIONS, TWO_ARG_FUNCTIONS};
use crate::math::std_dev;
use crate::serialization::deserializable::Deserializable;
use crate::serialization::serializator::Serializator;
use crate::serialization::Serializable;
use rand::{prelude::SliceRandom, Rng};
use std::collections::HashSet;
use std::{borrow::BorrowMut, ops::Deref, ops::Mul};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Weight(f32);

impl Serializable for Weight {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.0);
    }
}

impl Deserializable for Weight {
    fn deserialize(s: &mut Serializator) -> Result<Weight, String> {
        let v = s.next_token()?;
        Ok(Weight::from(
            v.parse().map_err(|e| format!("Unable to parse {} to Weight: {}", v, e))?,
        ))
    }
}

impl Weight {
    pub fn generate() -> Self {
        Weight { 0: GET_RNG().gen_range(-1.618..1.618) }
    }

    #[must_use]
    pub fn from(v: f32) -> Weight {
        Weight { 0: v }
    }
}

impl Mul<f32> for &Weight {
    type Output = f32;

    fn mul(self, rhs: f32) -> Self::Output {
        self.0 * rhs
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Weighted {
    w: Weight,
    n: Box<Node>,
}

impl Weighted {
    #[must_use]
    pub fn from(n: Node) -> Self {
        Weighted { w: Weight(1.0), n: Box::new(n) }
    }

    pub fn change_weight(&mut self, rate: f32) {
        let new = self.w.0 * rate;
        if !new.is_nan() {
            self.w.0 = new;
        }
    }

    pub fn mutate(&mut self, input_shape: &InputShape, forbidden_cols: &[usize]) {
        self.n.mutate(input_shape, forbidden_cols);
    }

    pub fn copy_internals(&mut self, n: Weighted) {
        self.w = n.w;
        self.n = n.n
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.n.node_count()
    }

    #[must_use]
    pub fn execute(&self, data: &Data<Vec<f32>>) -> Vec<f32> {
        let mut v = match self.n.deref() {
            Node::MathConstant(v) => vec![v.value(); data.get(0, 0).len()],
            Node::DataValue(r, c) => data.get(*r, *c).clone(),
            Node::StdDev(r, c) => std_dev(data.get(*r, *c)),
            Node::SingleArgFunction(f, n) => (f.fun)(n.execute(data)),
            Node::DoubleArgFunction(f, n1, n2) => (f.fun)(n1.execute(data), &n2.execute(data)),
        };
        v.iter_mut().for_each(|v| {
            *v = &self.w * *v;
        });
        v
    }

    pub fn new(
        current_depth: usize,
        input_shape: &InputShape,
        branch_prob: f64,
        max_depth: usize,
        forbidden_cols: &[usize],
        data_prob: f64,
    ) -> Weighted {
        if current_depth == max_depth {
            Weighted::new_terminating_node(input_shape, forbidden_cols, data_prob)
        } else {
            Weighted::new_function_node(
                current_depth,
                input_shape,
                branch_prob,
                max_depth,
                forbidden_cols,
                data_prob,
            )
        }
    }

    fn new_function_node(
        current_depth: usize,
        input_shape: &InputShape,
        branch_prob: f64,
        max_depth: usize,
        forbidden_cols: &[usize],
        data_prob: f64,
    ) -> Weighted {
        let mut rng = GET_RNG();
        let current_depth = current_depth + 1;
        let wn = Weighted::new(
            current_depth,
            input_shape,
            branch_prob,
            max_depth,
            forbidden_cols,
            data_prob,
        );
        if rng.gen_bool(branch_prob) {
            let wn2 = Weighted::new(
                current_depth,
                input_shape,
                branch_prob,
                max_depth,
                forbidden_cols,
                data_prob,
            );
            let n =
                Node::DoubleArgFunction(TWO_ARG_FUNCTIONS.choose(&mut rng).unwrap(), wn, wn2);
            let w = Weight::generate();
            Weighted { w, n: Box::new(n) }
        } else {
            let n = Node::SingleArgFunction(ONE_ARG_FUNCTIONS.choose(&mut rng).unwrap(), wn);
            let w = Weight::generate();
            Weighted { w, n: Box::new(n) }
        }
    }

    fn new_terminating_node(
        input_shape: &InputShape,
        forbidden_cols: &[usize],
        data_prob: f64,
    ) -> Weighted {
        if GET_RNG().gen_bool(data_prob) {
            Weighted::new_data_value_node(input_shape, forbidden_cols)
        } else {
            let n = Node::MathConstant(MATH_CONSTANTS.choose(&mut GET_RNG()).unwrap());
            let w = Weight::generate();
            Weighted { w, n: Box::new(n) }
        }
    }

    fn new_data_value_node(input_shape: &InputShape, forbidden_cols: &[usize]) -> Weighted {
        let (row, column) = input_shape.random_row_column(forbidden_cols);
        let n = if GET_RNG().gen_bool(0.95) {
            Node::DataValue(row, column)
        } else {
            Node::StdDev(row, column)
        };
        let w = Weight::generate();
        Weighted { w, n: Box::new(n) }
    }

    pub fn get_used_columns(&self) -> HashSet<usize> {
        let mut columns = HashSet::new();
        let mut node_queue = Vec::with_capacity(1024);
        node_queue.push(self.n.deref());
        while !node_queue.is_empty() {
            let next_node = node_queue.remove(0);
            match next_node {
                Node::DoubleArgFunction(_, n1, n2) => {
                    node_queue.push(n1.n.deref());
                    node_queue.push(n2.n.deref());
                }
                Node::SingleArgFunction(_, n) => {
                    node_queue.push(n.n.deref());
                }
                Node::MathConstant(_) => {}
                Node::DataValue(_, c) | Node::StdDev(_, c) => {
                    columns.insert(*c);
                }
            }
        }
        columns
    }

    pub fn take_node(self, id: usize) -> Weighted {
        let mut count = id;
        let mut node_queue = Vec::with_capacity(1024);
        let mut next_node = self;
        while count > 0 {
            count -= 1;
            match *next_node.n {
                Node::DoubleArgFunction(_, n1, n2) => {
                    node_queue.push(n2);
                    next_node = n1;
                }
                Node::SingleArgFunction(_, n) => {
                    next_node = n;
                }
                Node::DataValue(_, _) | Node::MathConstant(_) | Node::StdDev(_, _) => {
                    if !node_queue.is_empty() {
                        next_node = node_queue.remove(node_queue.len() - 1);
                    }
                }
            }
        }
        next_node
    }

    pub fn select_node_mut(&mut self, id: usize, total: usize) -> &mut Weighted {
        let mut count = id;
        let mut node_queue = Vec::with_capacity(total);
        let mut next_node = self;
        while count > 0 {
            count -= 1;
            match *next_node.n {
                Node::DoubleArgFunction(_, ref mut n1, ref mut n2) => {
                    node_queue.push(n2.borrow_mut());
                    next_node = n1.borrow_mut();
                }
                Node::SingleArgFunction(_, ref mut n) => {
                    next_node = n.borrow_mut();
                }
                Node::DataValue(_, _) | Node::MathConstant(_) | Node::StdDev(_, _) => {
                    if !node_queue.is_empty() {
                        next_node = node_queue.remove(node_queue.len() - 1);
                    }
                }
            }
        }
        next_node
    }
}

impl Serializable for Weighted {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.w, self.n.deref()])
    }
}

impl Deserializable for Weighted {
    fn deserialize(s: &mut Serializator) -> Result<Weighted, String> {
        let w = Weight::deserialize(s)?;
        let n = Box::new(Node::deserialize(s)?);
        Ok(Weighted { w, n })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Node {
    DataValue(usize, usize),
    StdDev(usize, usize),
    MathConstant(&'static MathConst),
    SingleArgFunction(&'static SingleArgFunction, Weighted),
    DoubleArgFunction(&'static DoubleArgFunction, Weighted, Weighted),
}

impl Serializable for Node {
    fn serialize(&self, s: &mut Serializator) {
        match self {
            Node::DataValue(row, column) => s.add_items(&[&"DataValue".to_owned(), row, column]),
            Node::StdDev(row, column) => s.add_items(&[&"StdDev".to_owned(), row, column]),
            Node::MathConstant(constant) => {
                s.add_items(&[&"Constant".to_owned(), constant.to_owned()])
            }
            Node::SingleArgFunction(fun, node) => {
                s.add_items(&[&"OneArgNode".to_owned(), fun.to_owned(), node.deref()])
            }
            Node::DoubleArgFunction(fun, n1, n2) => {
                s.add_items(&[&"TwoArgNode".to_owned(), fun.to_owned(), n1.deref(), n2.deref()])
            }
        }
    }
}

impl Deserializable for Node {
    fn deserialize(s: &mut Serializator) -> Result<Node, String> {
        let node_type = s.next_token()?;
        match node_type.as_str() {
            "DataValue" => {
                let row = usize::deserialize(s)?;
                let column = usize::deserialize(s)?;
                Ok(Node::DataValue(row, column))
            }
            "StdDev" => {
                let row = usize::deserialize(s)?;
                let column = usize::deserialize(s)?;
                Ok(Node::StdDev(row, column))
            }
            "Constant" => {
                let c = Deserializable::deserialize(s)?;
                Ok(Node::MathConstant(c))
            }
            "OneArgNode" => {
                let fun = Deserializable::deserialize(s)?;
                let node = Weighted::deserialize(s)?;
                Ok(Node::SingleArgFunction(fun, node))
            }
            "TwoArgNode" => {
                let fun = Deserializable::deserialize(s)?;
                let n1 = Weighted::deserialize(s)?;
                let n2 = Weighted::deserialize(s)?;
                Ok(Node::DoubleArgFunction(fun, n1, n2))
            }
            _ => Err(format!("Invalid node type {}", node_type)),
        }
    }
}

impl Node {
    #[must_use]
    pub fn one_arg_node(fun: &'static SingleArgFunction, wn: Weighted) -> Node {
        Node::SingleArgFunction(fun, wn)
    }

    #[must_use]
    pub fn two_arg_node(fun: &'static DoubleArgFunction, wn1: Weighted, wn2: Weighted) -> Node {
        Node::DoubleArgFunction(fun, wn1, wn2)
    }

    pub fn set_branch(&mut self, wn: &Weighted) {
        match self {
            Node::SingleArgFunction(_, ref mut n) | Node::DoubleArgFunction(_, _, ref mut n) => {
                *n = wn.clone()
            }
            _ => (),
        }
    }

    pub fn mutate(&mut self, input_shape: &InputShape, forbidden_cols: &[usize]) {
        let mut rng = GET_RNG();
        match self {
            Node::SingleArgFunction(ref mut f, _) => {
                *f = &ONE_ARG_FUNCTIONS.choose(&mut rng).unwrap()
            }
            Node::DoubleArgFunction(ref mut f, _, _) => {
                *f = &TWO_ARG_FUNCTIONS.choose(&mut rng).unwrap()
            }
            Node::MathConstant(ref mut c) => *c = &MATH_CONSTANTS.choose(&mut rng).unwrap(),
            Node::DataValue(ref mut row, ref mut column)
            | Node::StdDev(ref mut row, ref mut column) => {
                let (r, c) = input_shape.random_row_column(forbidden_cols);
                *row = r;
                *column = c;
            }
        }
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        1 + match self {
            Node::MathConstant(_) | Node::DataValue(_, _) | Node::StdDev(_, _) => 0,
            Node::SingleArgFunction(_, n) => n.n.node_count(),
            Node::DoubleArgFunction(_, n1, n2) => n1.n.node_count() + n2.n.node_count(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::exec::node::Weight;
    use crate::serialization::serializator::test::test_serialization;

    #[test]
    fn serialize_weight() {
        for _ in 0..1_000 {
            test_serialization(Weight::generate());
        }
    }
}
