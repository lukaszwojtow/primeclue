<template>
  <div>
    <div>
      <div style="text-align: left; font-size: large">
        Training classifier for data: {{ this.$props.dataName }}
      </div>
    </div>
    <div style="text-align: left">
      <div style="display: inline-flex">
        <div>
          <div style="text-align: left">
            <div style="display: inline-flex; padding-top: 15px;">
              <div style="padding-top: 15px; width: 170px">
                Classifier name:
              </div>
              <el-input style="width: 300px; margin-left: 10px" v-model="classifierName"/>
            </div>
          </div>


          <div style="text-align: left; padding-top: 15px">
            <div style="display: inline-flex">
              <div style="text-align: left; margin-top: 15px; width: 170px">
                Timeout (minutes):
              </div>
              <el-input-number style="margin-left: 10px; width: 300px" v-model="timeout" :min="1" :step="10"/>
            </div>
          </div>

          <div style="text-align: left; padding-top: 15px">
            <div style="display: inline-flex">
              <div style="text-align: left; margin-top: 15px; width: 170px">
                Size:
              </div>
              <el-input-number style="width: 300px; margin-left: 10px" v-model="size" :min="2"
                               :step="10"/>
            </div>
          </div>
          <div>
            <div style="display: inline-flex; padding-top: 15px; text-align: left">
              <div style="margin-top: 15px; width: 170px">
                Training objective:
              </div>
              <el-select style="width: 300px; margin-left: 10px" v-model="trainingObjective"
                         placeholder="Select training objective">
                <el-option
                    v-for="name in possibleTrainingObjectives"
                    :key="name"
                    :label="name"
                    :value="name">
                </el-option>
              </el-select>
            </div>
          </div>
          <div style="text-align: left; padding-top: 15px">
            <div style="display: inline-flex">
              <div style="text-align: left; margin-top: 15px; width: 170px">
                Forbidden columns:
              </div>
              <el-input style="width: 300px; margin-left: 10px" v-model="forbiddenColumns"/>
            </div>
          </div>
          <div style="text-align: left; padding-top: 15px">
            <div style="display: inline-flex">
              <div style="text-align: left; margin-top: 15px; width: 170px">
                Shuffle data:
              </div>
              <el-checkbox style="margin-top: 10px; margin-left: 10px" v-model="shuffleData"/>
            </div>
          </div>

          <div style="text-align: left; padding-top: 15px">
            <div style="display: inline-flex">
              <div style="text-align: left; margin-top: 15px; width: 170px">
                Keep unseen data:
              </div>
              <el-checkbox style="margin-top: 10px; margin-left: 10px" v-model="keepUnseenData"/>
            </div>
          </div>

          <div style="text-align: left; padding-top: 15px">
            <div style="display: inline-flex">
              <div style="text-align: left">
                <el-button style="width: 170px; margin-bottom: 15px" @click="startStop">{{ stopStartButtonText }}
                </el-button>
              </div>
              <div v-if="jobInProgress" style="margin-top: 15px; margin-left: 10px">
                Executing training with id: {{ jobId }}
              </div>
            </div>
          </div>
        </div>
        <div>
          <div style="display: inline-flex; text-align: left; margin-left: 20px">
            <div style="padding-top: 15px; width: 470px">
              <div v-if="trainingObjective === 'Cost'" id="rewards">
                <el-checkbox :border=true size="large" style="width: 470px" v-model="overrideRewards">
                  Override rewards / penalties
                </el-checkbox>
                <div v-if="overrideRewards">
                  <div style="display: inline-flex; padding-top: 15px">
                    <div style="width: 170px; margin-top: 15px">
                      Reward
                    </div>
                    <el-input-number style="width: 300px" v-model="reward" :min="0"/>
                  </div>
                  <div style="display: inline-flex; padding-top: 15px">
                    <div style="width: 170px; margin-top: 15px">
                      Penalty
                    </div>
                    <el-input-number style="width: 300px" v-model="penalty" :max="0"/>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <el-table
        :data="progress_data"
        :highlight-current-row="true"
        :border=true
        empty-text="No data available"
        style="width: 100%">
      <el-table-column
          prop="time"
          label="Time"
          width="300"/>
      <el-table-column
          prop="generation"
          label="Generation"
          width="150"/>
      <el-table-column
          prop="node_count"
          label="Node count"
          width="150"/>
      <el-table-column
          label="Scores">
        <el-table-column
            prop="training_score"
            label="Training score"
            />
        <el-table-column
            label="Unseen data">
          <el-table-column
            prop="test_auc"
            label="Auc"
            />
          <el-table-column
              prop="test_accuracy"
              label="%"
              />
          <el-table-column
              prop="test_cost"
              label="Cost"
              />
          <el-table-column
              prop="test_labels"
              label="Guesses"
              />
          <el-table-column
              prop="accuracy_labels"
              label="Label's accuracy"
          />
        </el-table-column>
      </el-table-column>
    </el-table>
  </div>
</template>
<script>

import {http} from "@/http";
import {date_str} from "../basic/DateString";
import {MessageBox} from 'element-ui';
import ulog from 'ulog'

export default {
  name: "classifierCreate",
  props: ['dataName'],
  log: ulog('classifier_create'),
  data() {
    return {
      showDoneWindow: true,
      trainingObjective: "Auc",
      possibleTrainingObjectives: ["Accuracy", "Cost", "Auc"],
      classifierName: "",
      overrideRewards: false,
      forbiddenColumns: "",
      shuffleData: true,
      keepUnseenData: true,
      reward: 1.0,
      penalty: -1.0,
      progress_data: [],
      jobId: "(pending)",
      jobInProgress: false,
      timeout: 60,
      size: 10,
      stopStartButtonText: "Start",
    }
  },
  methods: {
    startStop() {
      if (!this.jobInProgress) {
        this.start();
        this.showDoneWindow = true;
        this.stopStartButtonText = "Stop";
      } else {
        this.stop();
        this.showDoneWindow = false;
        this.stopStartButtonText = "Start";
      }
    },
    start() {
      let request = this.buildRequest();
      this.progress_data = [];
      this.jobInProgress = true;
      http.postAsync("/classifier/create", request, (status, job_id) => {
        this.jobId = job_id;
        if (status.Progress !== undefined) {
          let tick = JSON.parse(status.Progress[1]);
          if (this.progress_data.length > 0 && this.progress_data[this.progress_data.length - 1].generation === tick.stats.generation) {
            return;
          }
          let now = date_str.str();
          let p = {
            time: now,
            generation: tick.stats.generation,
            node_count: tick.stats.node_count,
            training_score: tick.stats.training_score.toFixed(3),
            test_auc: tick.classifier_score.auc.toFixed(2),
            test_accuracy: tick.classifier_score.accuracy.toFixed(3),
            test_cost: tick.classifier_score.cost.toFixed(1),
            test_labels: JSON.stringify(tick.classifier_score.label_count_map)
                .replaceAll('{','')
                .replaceAll('}','')
                .replaceAll(',', ' ')
                .replaceAll('"','')
                .replaceAll(':', ': '),
            accuracy_labels: JSON.stringify(tick.classifier_score.label_accuracy_map)
                .replaceAll('{','')
                .replaceAll('}','')
                .replaceAll(',', ' ')
                .replaceAll('"','')
                .replaceAll(':', ': '),
          };
          this.progress_data.push(p);
        }
      }, (e) => {
        http.generalErrorHandler(e);
        this.stopStartButtonText = "Start";
        this.jobInProgress = false;
      }, (r) => {
        if (this.showDoneWindow) {
          MessageBox.alert(r, {
            title: "Done",
            type: "info",
            confirmButtonText: 'Close',
          });
        }
        this.stopStartButtonText = "Start";
        this.jobInProgress = false;
      });
    },
    stop() {
      if (!this.jobInProgress) {
        MessageBox.alert("Job not started yet", {
          title: "Error",
          type: "error",
          confirmButtonText: 'Close',
        })
      } else {
        http.put(`/job/${this.jobId}/terminate`)
            .catch(e => http.generalErrorHandler(e));
        this.jobInProgress = false;
      }
    },
    buildRequest() {
      return {
        data_name: this.$props.dataName,
        classifier_name: this.classifierName,
        training_objective: this.trainingObjective,
        override_rewards: this.overrideRewards,
        forbidden_columns: this.forbiddenColumns,
        shuffle_data: this.shuffleData,
        keep_unseen_data: this.keepUnseenData,
        rewards: {
          reward: this.reward,
          penalty: this.penalty,
        },
        timeout: this.timeout,
        size: this.size,
      }
    },
  },
  mounted() {
    this.classifierName = `${this.$props.dataName}_classifier_${date_str.now()}`;
  },
}
</script>

<style scoped>

</style>