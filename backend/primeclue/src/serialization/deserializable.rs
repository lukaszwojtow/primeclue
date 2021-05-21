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

use crate::serialization::serializator::{Serializator, PRIMECLUE_SPACE_SUBSTITUTE};
use std::collections::HashMap;
use std::hash::Hash;

pub trait Deserializable {
    fn deserialize(s: &mut Serializator) -> Result<Self, String>
    where
        Self: Sized;
}

impl Deserializable for usize {
    fn deserialize(s: &mut Serializator) -> Result<usize, String> {
        let v = s.next_token()?;
        v.parse().map_err(|e| format!("Unable to parse '{}': {:?}", v, e))
    }
}

impl Deserializable for u16 {
    fn deserialize(s: &mut Serializator) -> Result<u16, String> {
        let v = s.next_token()?;
        v.parse().map_err(|e| format!("Unable to parse '{}': {:?}", v, e))
    }
}

impl Deserializable for String {
    fn deserialize(s: &mut Serializator) -> Result<String, String> {
        match s.next_token() {
            Ok(s) => Ok(s.replace(PRIMECLUE_SPACE_SUBSTITUTE, " ")),
            Err(e) => Err(e.to_string()), // TODO test
        }
    }
}

impl Deserializable for f32 {
    fn deserialize(s: &mut Serializator) -> Result<f32, String> {
        let v = s.next_token()?;
        v.parse().map_err(|e| format!("Unable to parse '{}': {:?}", v, e))
    }
}

impl Deserializable for bool {
    // TODO test
    fn deserialize(s: &mut Serializator) -> Result<bool, String> {
        let v = s.next_token()?;
        v.parse().map_err(|e| format!("Unable to parse '{}': {:?}", v, e))
    }
}

impl<T: Deserializable> Deserializable for Option<T> {
    fn deserialize(s: &mut Serializator) -> Result<Option<T>, String> {
        let t = s.next_token()?;
        match t.as_ref() {
            "None" => Ok(None),
            "Some" => {
                let v = T::deserialize(s)?;
                Ok(Some(v)) // TODO test
            }
            _ => Err(format!("Invalid token when deserializing Option: '{}'", t)),
        }
    }
}

impl<T: Deserializable, U: Deserializable> Deserializable for (T, U) {
    fn deserialize(s: &mut Serializator) -> Result<Self, String>
    where
        Self: Sized,
    {
        let first = T::deserialize(s)?;
        let second = U::deserialize(s)?;
        Ok((first, second))
    }
}

impl<T: Deserializable> Deserializable for Vec<T> {
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        let len = usize::deserialize(s)?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::deserialize(s)?);
        }
        Ok(vec)
    }
}

impl<
        K: Deserializable + Eq + Hash,
        V: Deserializable,
        S: ::std::hash::BuildHasher + Default,
    > Deserializable for HashMap<K, V, S>
{
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        let len = usize::deserialize(s)?;
        let mut map = HashMap::default();
        for _ in 0..len {
            let k = K::deserialize(s)?;
            let v = V::deserialize(s)?;
            map.insert(k, v);
        }
        Ok(map)
    }
}
