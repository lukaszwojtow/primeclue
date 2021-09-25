<template>
    <div class="all">
        <async-progress title="Importing data" ref="importProgressWindow"/>
        <div style="display: inline-block">
            <div style="display: inline-flex; padding-top: 15px">
                <el-upload :auto-upload="false" action="" :on-change="fileSelected" :show-file-list="false">
                    <el-button style="width: 170px;" size="small" type="primary" :loading="fileLoading">
                        {{fileLoadButtonText}}
                    </el-button>
                </el-upload>
                <div style="white-space: nowrap; margin-left: 10px; margin-top: 10px">
                    Field separator:
                </div>
                <el-input size="small" style="margin-left: 10px; width: 70px" v-model="fieldSeparator"
                          @input="rewriteContent" type="text"
                          maxlength="1"/>
                <el-checkbox style="margin-left: 10px; margin-top: 10px" v-model="firstRowIsHeader">First row as
                    header
                </el-checkbox>
            </div>

            <div>
                <div style="display: inline-flex; padding-top: 15px">
                    <div style="width: 170px">
                      <div style="padding-top: 15px; width: 170px">
                        Class column:
                      </div>
                    </div>
                    <el-input-number size="small" style="width: 187px; margin-left: 10px"
                                   v-model="classColumn" :min="1" :max="columns" @change="unmarkClassColumn"/>
                    <el-button size="small" :disabled="!fileChosen" style="margin-left: 10px; width: 145px" type="large"
                               @click="getClasses">Get classes
                    </el-button>
                </div>
            </div>
            <div>
                <div style="display: inline-flex; padding-top: 15px">
                    <div style="padding-top: 15px; width: 170px">
                        Rows per set:
                    </div>
                    <el-input-number style="width: 187px; margin-left: 10px" size="small" v-model="rowsPerSet"
                                     :min="1"/>
                </div>
            </div>
            <div>
                <div style="display: inline-flex; padding-top: 15px">
                    <div style="white-space: nowrap; padding-top: 15px; width: 170px">
                        Data name:
                    </div>
                    <el-input size="small" style="margin-left: 10px; width: 188px" v-model="dataName" type="text"/>
                    <el-button size="small" :disabled="!fileChosen" style="margin-left: 10px; width: 145px" type="large"
                               @click="save">Save
                    </el-button>
                </div>
            </div>
        </div>

        <div style="display: inline-block; vertical-align: top">
            <div style="margin-left: 50px; padding-top: 15px">
                <div>
                    <el-checkbox style="margin-top: 5px; width: 305px" v-model="individualRewardsPenalties">
                        Individual reward / penalty columns
                    </el-checkbox>
                </div>
                <div style="display: inline-flex; margin-top: 10px">
                    <div style="padding-top: 20px; width: 110px">
                        Reward column:
                    </div>
                    <el-input-number size="small" :disabled="!individualRewardsPenalties"
                                     style="width: 125px; margin-left: 10px; margin-top: 10px"
                                     v-model="rewardColumn" :min="1" :max="columns"/>
                </div>
                <div>
                    <div style="display: inline-flex; margin-top: 5px">
                        <div style="padding-top: 20px; width: 110px">
                            Penalty column:
                        </div>
                        <el-input-number size="small" :disabled="!individualRewardsPenalties"
                                         style="width: 125px; margin-left: 10px; margin-top: 10px"
                                         v-model="penaltyColumn" :min="1" :max="columns"/>
                    </div>
                </div>
            </div>
        </div>

        <div style="overflow:auto; max-height:500px; margin-top: 10px">
            <table border="1" width="100%" v-html="this.table"></table>
        </div>
    </div>
</template>

<script>
    import {http} from '@/http';
    import {date_str} from "../basic/DateString";
    import AsyncProgress from "../basic/AsyncProgress";

    export default {
        name: "DataTable",
        components: {AsyncProgress},
        mounted() {
            this.dataName = `data_${date_str.now()}`;
        },
        methods: {
            fileSelected(file) {
                const reader = new FileReader();
                reader.onload = () => {
                    this.redrawTable(reader.result);
                    this.fileChosen = true;
                    setTimeout(() => {
                        this.fileLoading = false;
                        this.fileLoadButtonText = "Load file...";
                    });
                  this.unmarkClassColumn();
                };
                setTimeout(() => {
                    this.fileLoading = true;
                    this.fileLoadButtonText = "Loading";
                    reader.readAsText(file.raw);
                })
            },
            unmarkClassColumn() {
              let checkBox = document.getElementById(`importColumnCheckBox${this.classColumn-1}`);
              checkBox.checked = false;
            },
            redrawTable(content) {
                this.content = content;
                this.classes = [];
                this.rewriteContent();
            },
            firstRowHasNumbers(row) {
                let columns = row.trim().split(this.fieldSeparator);
                for (let c = 0; c < columns.length; c++) {
                    if (isNaN(parseFloat(columns[c].trim()))) {
                        return false;
                    }
                }
                return true;
            },
            rewriteContent() {
                let rows = this.content.split('\n');
                this.firstRowIsHeader = !this.firstRowHasNumbers(rows[0]);
                let trs = this.getTableHeader(rows[0].trim().split(this.fieldSeparator));
                for (let i = 0; i < rows.length; i++) {
                    if (rows[i] === "") {
                        continue;
                    }
                    let tds = "";
                    let cols = rows[i].trim().split(this.fieldSeparator);
                    this.columns = cols.length;
                    for (let c = 0; c < cols.length; c++) {
                        tds += `<td>${cols[c]}</td>`
                    }
                    if (this.classes[i] !== undefined) {
                        tds += `<td>${this.classes[i]}</td>`
                    }
                    if (cols.length > 0) {
                        trs += `<tr height="40">${tds}</tr>`;
                    }
                }
                this.table = trs;
            },
            getTableHeader(columns) {
                let header = `<tr height="40">`;
                let already_selected_columns = this.getImportColumns();
                for (let i = 0; i < columns.length; i++) {
                    let cb;
                    if (already_selected_columns.length > i && already_selected_columns[i] === false) {
                        cb = `<input type="checkbox" id="importColumnCheckBox${i}" title="Import column as input data">`;
                    } else {
                        cb = `<input type="checkbox" id="importColumnCheckBox${i}" title="Import column as input data" checked>`;
                    }
                    header += `<th><div>${i + 1}</div><div>${cb}</div></th>`;
                }
                header += "<th width='20%'>Outcomes</th>";
                return header + "</tr>";
            },
            getClasses() {
                let request = this.buildRequest();
                http.post("/data/classes", request)
                    .then(r => {
                        this.classes = r.data.classes;
                        this.rewriteContent();
                    })
                    .catch(e => http.generalErrorHandler(e))
            },
            save() {
                let request = this.buildRequest();
                http.postAsync("/data/import", request, (status) => {
                    this.$refs.importProgressWindow.show(status);
                }, (e) => http.generalErrorHandler(e));
            },
            buildRequest() {
                let expression = "";
                if (this.classProducer === "bool") {
                    expression = this.expression;
                }
                return {
                    content: this.content,
                    expression: expression,
                    class_column: this.classColumn,
                    separator: this.fieldSeparator,
                    ignore_first_row: this.firstRowIsHeader,
                    rows_per_set: this.rowsPerSet,
                    import_columns: this.getImportColumns(),
                    data_name: this.dataName,
                    custom_reward_penalty_columns: this.individualRewardsPenalties,
                    reward_column: this.rewardColumn,
                    penalty_column: this.penaltyColumn,
                };
            },
            getImportColumns() {
                let import_columns = [];
                for (let i = 0; i < 1024; i++) {
                    let check_box = document.getElementById(`importColumnCheckBox${i}`);
                    if (check_box === undefined || check_box === null) {
                        return import_columns;
                    } else {
                        import_columns.push(check_box.checked);
                    }
                }
            }
        },
        data() {
            return {
                fileLoading: false,
                fileLoadButtonText: "Load file ...",
                fileChosen: false,
                classColumn: 0,
                firstRowIsHeader: false,
                content: "",
                fieldSeparator: ",",
                dataName: "",
                table: "",
                classes: [],
                rowsPerSet: 1,
                individualRewardsPenalties: false,
                rewardColumn: 0,
                penaltyColumn: 0,
                columns: 0,
            }
        }
    }
</script>

<style scoped>
    table {
        border: 1px solid #ddd;
    }

    table {
        border-collapse: collapse;
        width: 100%;
    }

    td {
        padding: 15px;
        text-align: left;
    }

    th {
        border: 1px solid #ddd;
        padding: 15px;
        text-align: center;
    }

    .all {
        font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
        color: #606266;
    }

</style>