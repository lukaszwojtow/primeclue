<template>
    <div>
        <top-menu active="classifiers-window"/>
        <el-table
                :data="classifierList"
                :highlight-current-row="true"
                empty-text="No data available">
            <el-table-column
                    prop="name"
                    label="Name"
                    width="400">
            </el-table-column>
            <el-table-column
                    label="Operations"
                    width="200">
                <template v-slot="scope">
                  <el-popconfirm style="margin-right: 20pt" title="Are you sure to delete" cancel-button-text="Cancel" confirm-button-text="Delete" v-on:confirm="remove(scope.$index)">
                    <template v-slot:reference>
<el-button
                        
                        type="text"
                        size="small">
                      Remove
                    </el-button>
</template>
                  </el-popconfirm>
                    <el-button
                            @click.native.prevent="classify(scope.$index)"
                            type="text"
                            size="small">
                        Classify
                    </el-button>

                </template>
            </el-table-column>
        </el-table>
    </div>
</template>

<script>
    import {http} from '@/http';

    export default {
        name: "classifiersWindow",
        data() {
            return {
                classifierList: [],
            }
        },
        mounted() {
            this.getClassifierList();
        },
        methods: {
            getClassifierList() {
                http.getClassifierList()
                    .then(r => {
                        this.classifierList = [];
                        for (let i = 0; i < r.data.length; i++) {
                            let name = {name: r.data[i]};
                            this.classifierList.push(name);
                        }
                    })
                    .catch(e => http.generalErrorHandler(e));
            },
            remove(index) {
                http.post(`/classifier/remove/${this.classifierList[index].name}`)
                    .then(r => this.getClassifierList(r))
                    .catch(e => http.generalErrorHandler(e))
            },
            classify(index) {
                this.$router.push(
                    {name: 'classification', params: {classifierName: this.classifierList[index].name}}
                );
            }
        }
    }
</script>

<style scoped>

</style>