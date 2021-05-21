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

use crate::serialization::serializator::Serializator;
use std::collections::HashMap;
use std::hash::BuildHasher;

pub trait Serializable {
    fn serialize(&self, s: &mut Serializator);
}

impl Serializable for usize {
    fn serialize(&self, s: &mut Serializator) {
        s.add_string(format!("{}", self));
    }
}

impl Serializable for u16 {
    fn serialize(&self, s: &mut Serializator) {
        s.add_string(format!("{}", self));
    }
}

impl Serializable for String {
    fn serialize(&self, s: &mut Serializator) {
        s.add_string(self.to_owned());
    }
}

impl Serializable for f64 {
    fn serialize(&self, s: &mut Serializator) {
        s.add_string(format!("{}", self));
    }
}

impl Serializable for f32 {
    fn serialize(&self, s: &mut Serializator) {
        s.add_string(format!("{}", self));
    }
}

impl Serializable for bool {
    fn serialize(&self, s: &mut Serializator) {
        s.add_string(format!("{}", self));
    }
}

impl<T: Serializable, U: Serializable> Serializable for (T, U) {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.0);
        s.add(&self.1);
    }
}

impl<T: Serializable> Serializable for Option<T> {
    fn serialize(&self, s: &mut Serializator) {
        match self {
            None => s.add_str("None"),
            Some(v) => s.add_items(&[&"Some".to_owned(), v]),
        }
    }
}

impl<T: Serializable> Serializable for Vec<T> {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.len());
        for item in self {
            s.add(item);
        }
    }
}

impl<K: Serializable, V: Serializable, S: BuildHasher> Serializable for HashMap<K, V, S> {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.len());
        for (k, v) in self {
            s.add(k);
            s.add(v);
        }
    }
}
