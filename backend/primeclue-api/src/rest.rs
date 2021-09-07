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

use crate::classifier::{create, ClassifyRequest, CreateRequest};
use crate::data::{classes, import};
use crate::executor::{Executor, Job, JobId, Status, Termination};
use crate::{classifier, data};
use actix_cors::Cors;
use actix_web::{http, web, App, HttpResponse, HttpServer};
use primeclue::data::importer::ClassRequest;
use serde::Serialize;
use std::sync::mpsc::channel;
use std::sync::Mutex;

pub(crate) const SERVER_ADDR: &str = "0.0.0.0";
pub(crate) const SERVER_PORT: u16 = 8180;

#[derive(Serialize)]
struct StatusResponse {
    status: Status,
}

#[allow(clippy::needless_pass_by_value)]
fn job_status_handler(path: web::Path<JobId>, data: web::Data<Mutex<Executor>>) -> HttpResponse {
    let id = path.into_inner();
    let mut executor = data.lock().unwrap();
    let status = executor.status(id);
    match status {
        Some(status) => match status {
            Status::Error(err) => {
                HttpResponse::InternalServerError().body(format!("Error: {}", err))
            }
            _ => HttpResponse::Ok().json(StatusResponse { status }),
        },
        None => HttpResponse::NotFound().body(format!("Error: JobId {} not found", id)),
    }
}

#[allow(clippy::needless_pass_by_value)]
fn job_terminate_handler(
    path: web::Path<JobId>,
    data: web::Data<Mutex<Executor>>,
) -> HttpResponse {
    let id = path.into_inner();
    let executor = data.lock().unwrap();
    match executor.terminate(id) {
        Ok(_) => HttpResponse::Ok().finish(),
        Err(error) => HttpResponse::BadRequest().body(format!("Error: {}", error)),
    }
}

fn data_remove_handler(path: web::Path<String>) -> HttpResponse {
    let name = path.into_inner();
    match data::remove(&name) {
        Ok(_) => HttpResponse::Ok().finish(),
        Err(error) => HttpResponse::InternalServerError().body(format!("Error: {}", error)),
    }
}

fn data_list_handler() -> HttpResponse {
    match data::list() {
        Ok(list) => HttpResponse::Ok().json(list),
        Err(error) => HttpResponse::InternalServerError().body(format!("Error: {}", error)),
    }
}

fn classifier_list_handler() -> HttpResponse {
    match classifier::list() {
        Ok(list) => HttpResponse::Ok().json(list),
        Err(error) => HttpResponse::InternalServerError().body(format!("Error: {}", error)),
    }
}

fn classifier_remove_handler(path: web::Path<String>) -> HttpResponse {
    let name = path.into_inner();
    match classifier::remove(&name) {
        Ok(_) => HttpResponse::Ok().finish(),
        Err(error) => HttpResponse::InternalServerError().body(format!("Error: {}", error)),
    }
}

fn data_classes_handler(r: web::Json<ClassRequest>) -> HttpResponse {
    match classes(&r.into_inner()) {
        Ok(classes) => HttpResponse::Ok().json(classes),
        Err(error) => HttpResponse::BadRequest().body(format!("Error: {}", error)),
    }
}

#[allow(clippy::needless_pass_by_value)]
fn data_import_handler(
    r: web::Json<ClassRequest>,
    data: web::Data<Mutex<Executor>>,
) -> HttpResponse {
    let mut executor = data.lock().unwrap();
    let (id, callback) = executor.prepare_new_job();
    let job = Job::new(id, Box::new(move || import(r.into_inner(), &callback)));
    executor.submit(job, None);
    id_ok_response(id)
}

fn id_ok_response(id: u64) -> HttpResponse {
    HttpResponse::Ok().body(format!("{}", id))
}

fn classifier_classify_handler(r: web::Json<ClassifyRequest>) -> HttpResponse {
    let request = r.into_inner();
    match request.classify() {
        Ok(results) => HttpResponse::Ok().body(results),
        Err(error) => HttpResponse::InternalServerError().body(format!("Error: {}", error)),
    }
}

#[allow(clippy::needless_pass_by_value)]
fn classifier_create_handler(
    r: web::Json<CreateRequest>,
    data: web::Data<Mutex<Executor>>,
) -> HttpResponse {
    let mut executor = data.lock().unwrap();
    let (id, callback) = executor.prepare_new_job();
    let (terminator_sender, terminator) = channel::<Termination>();
    let job = Job::new(id, Box::new(move || create(&r.into_inner(), &callback, &terminator)));
    executor.submit(job, Some(terminator_sender));
    id_ok_response(id)
}

pub(crate) async fn start_web() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(Cors::permissive())
            .app_data(web::Data::new(Mutex::new(Executor::create())))
            .app_data(web::PayloadConfig::new(1024 * 1024 * 16))
            .route("/data/classes", web::to(data_classes_handler).method(http::Method::POST))
            .route("/data/import", web::to(data_import_handler).method(http::Method::POST))
            .route("/data/list", web::to(data_list_handler).method(http::Method::GET))
            .route(
                "/data/remove/{name}",
                web::to(data_remove_handler).method(http::Method::POST),
            )
            .route(
                "/classifier/classify",
                web::to(classifier_classify_handler).method(http::Method::POST),
            )
            .route(
                "/classifier/create",
                web::to(classifier_create_handler).method(http::Method::POST),
            )
            .route(
                "/classifier/list",
                web::to(classifier_list_handler).method(http::Method::GET),
            )
            .route(
                "/classifier/remove/{name}",
                web::to(classifier_remove_handler).method(http::Method::POST),
            )
            .route("/job/{id}/status", web::to(job_status_handler).method(http::Method::GET))
            .route(
                "/job/{id}/terminate",
                web::to(job_terminate_handler).method(http::Method::PUT),
            )
    })
    .bind((SERVER_ADDR, SERVER_PORT))?
    .run()
    .await
}
