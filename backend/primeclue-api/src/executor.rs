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

use primeclue::error::PrimeclueErr;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use threadpool::ThreadPool;

pub(crate) type JobId = u64;
pub(crate) type StatusCallback = Box<dyn Fn(Status) + Send>;
pub(crate) type Task = Box<dyn FnOnce() -> Result<String, PrimeclueErr> + Send>;

#[derive(Debug)]
pub(crate) struct StatusUpdate {
    id: JobId,
    status: Status,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) enum Status {
    Ok(String),
    Error(PrimeclueErr),
    Progress(f64, String),
    Pending,
}

pub(crate) enum Termination {
    Cancel,
}

impl Status {
    pub(crate) fn is_final(&self) -> bool {
        match self {
            Status::Ok(_) | Status::Error(_) => true,
            Status::Progress(_, _) | Status::Pending => false,
        }
    }
}

pub(crate) struct Job {
    id: JobId,
    task: Task,
}

impl Job {
    pub(crate) fn new(id: JobId, task: Task) -> Job {
        Job { id, task }
    }
}

pub(crate) struct Executor {
    // TODO test
    next_id: u64,
    job_sender: Sender<Job>,
    status_sender: StatusSender,
    status_map: Arc<Mutex<HashMap<JobId, Status>>>,
    terminators: Arc<Mutex<HashMap<JobId, Sender<Termination>>>>,
}

impl Executor {
    pub(crate) fn create() -> Self {
        let (job_sender, job_receiver) = channel::<Job>();
        let (status_sender, status_receiver) = channel::<StatusUpdate>();
        Executor::start_job_receiver_thread(job_receiver, &status_sender);
        let status_map = Arc::new(Mutex::new(HashMap::new()));
        let terminators = Arc::new(Mutex::new(HashMap::new()));
        Executor::start_status_receiver_thread(status_receiver, &status_map, &terminators);
        let status_sender = StatusSender::new(status_sender);
        Executor { next_id: 1_u64, job_sender, status_sender, status_map, terminators }
    }

    pub(crate) fn submit(&self, job: Job, terminator: Option<Sender<Termination>>) {
        if let Some(t) = terminator {
            let mut r = self.terminators.lock().unwrap();
            r.insert(job.id, t);
        }
        self.job_sender.send(job).unwrap();
    }

    pub(crate) fn prepare_new_job(&mut self) -> (JobId, StatusCallback) {
        let job_id = self.next_id;
        self.status_map.lock().unwrap().insert(job_id, Status::Pending);
        let sender = self.status_sender.clone();
        self.next_id += 1;
        (job_id, Box::new(move |status| sender.send(status, job_id)))
    }

    pub(crate) fn status(&mut self, job_id: JobId) -> Option<Status> {
        let mut map = self.status_map.lock().unwrap();
        if let Some(status) = map.get(&job_id) {
            return if status.is_final() { map.remove(&job_id) } else { Some(status.clone()) };
        }
        None
    }

    pub(crate) fn terminate(&self, job_id: JobId) -> Result<(), PrimeclueErr> {
        let terminators = self.terminators.lock().unwrap();
        terminators
            .get(&job_id)
            .map(|s| s.send(Termination::Cancel).unwrap())
            .ok_or_else(|| PrimeclueErr::from(format!("Job {} not found", job_id)))
    }

    fn start_status_receiver_thread(
        status_receiver: Receiver<StatusUpdate>,
        status_map: &Arc<Mutex<HashMap<JobId, Status>>>,
        terminators_map: &Arc<Mutex<HashMap<JobId, Sender<Termination>>>>,
    ) {
        let receiver_map = status_map.clone();
        let thread_terminators = terminators_map.clone();
        thread::spawn(move || {
            for status in status_receiver.iter() {
                if status.status.is_final() {
                    thread_terminators.lock().unwrap().remove(&status.id);
                }
                receiver_map.lock().unwrap().insert(status.id, status.status);
            }
        });
    }

    fn start_job_receiver_thread(
        job_receiver: Receiver<Job>,
        executor_status_sender: &Sender<StatusUpdate>,
    ) {
        let executor_status_sender = executor_status_sender.clone();
        thread::spawn(move || {
            let pool = ThreadPool::default();
            for job in job_receiver.iter() {
                let pool_status_sender = executor_status_sender.clone();
                pool.execute(move || {
                    let start = std::time::Instant::now();
                    println!("Starting job {}", job.id);
                    let status = match (job.task)() {
                        Ok(m) => Status::Ok(m),
                        Err(err) => Status::Error(err),
                    };
                    println!(
                        "Job {} done in {:?}",
                        job.id,
                        std::time::Instant::now().duration_since(start)
                    );
                    let update = StatusUpdate { id: job.id, status };
                    pool_status_sender.send(update).unwrap();
                })
            }
        });
    }
}

#[derive(Clone)]
struct StatusSender {
    status_sender: Sender<StatusUpdate>,
}

impl StatusSender {
    fn new(status_sender: Sender<StatusUpdate>) -> Self {
        StatusSender { status_sender }
    }

    fn send(&self, s: Status, job_id: JobId) {
        match s {
            Status::Ok(m) => self.send_ok(job_id, m),
            Status::Error(e) => self.send_error(job_id, e),
            Status::Progress(v, m) => self.send_progress(job_id, v, m),
            Status::Pending => {}
        };
    }

    fn send_error(&self, job_id: JobId, error: PrimeclueErr) {
        self.status_sender
            .send(StatusUpdate { id: job_id, status: Status::Error(error) })
            .unwrap();
    }

    fn send_progress(&self, job_id: JobId, value: f64, message: String) {
        self.status_sender
            .send(StatusUpdate { id: job_id, status: Status::Progress(value, message) })
            .unwrap();
    }

    fn send_ok(&self, job_id: JobId, message: String) {
        self.status_sender
            .send(StatusUpdate { id: job_id, status: Status::Ok(message) })
            .unwrap();
    }
}
