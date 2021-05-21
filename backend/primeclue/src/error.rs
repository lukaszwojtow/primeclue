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

use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct PrimeclueErr {
    err: String,
}

impl Error for PrimeclueErr {}

impl PrimeclueErr {
    pub fn result<T>(err: String) -> Result<T, PrimeclueErr> {
        Err(PrimeclueErr::from(err))
    }
}

impl From<String> for PrimeclueErr {
    fn from(err: String) -> Self {
        PrimeclueErr { err }
    }
}

// TODO test
impl From<std::io::Error> for PrimeclueErr {
    fn from(err: std::io::Error) -> Self {
        PrimeclueErr::from(err.to_string())
    }
}

// TODO test
impl Display for PrimeclueErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.err)
    }
}
