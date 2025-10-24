<template>
    <div>
        <div style="text-align: left; margin-top: 30px; margin-bottom: 30px">
            <el-button @click="openImportDataWindow">Import new data...</el-button>
        </div>


        <el-table
                :data="dataList"
                :highlight-current-row="true"
                empty-text="No data available">
            <el-table-column
                    prop="name"
                    label="Data name"
                    width="200">
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
                            @click.native.prevent="train(scope.$index)"
                            type="text"
                            size="small">
                        Train classifier
                    </el-button>
                </template>
            </el-table-column>
        </el-table>
    </div>
</template>

<script>
    import {http} from '@/http';

    export default {
        name: "DataList",
        data() {
            return {
                dataList: [],
            }
        },
        mounted() {
            this.getList();
        },
        methods: {
            openImportDataWindow() {
                this.$router.push('data-import')
            },

            getList() {
                http.getDataList()
                    .then(r => {
                        this.dataList = [];
                        for (let i = 0; i < r.data.length; i++) {
                            let name = {name: r.data[i]};
                            this.dataList.push(name);
                        }
                    })
                    .catch(e => http.generalErrorHandler(e))
            },

            remove(id) {
                let name = this.dataList[id].name;
                http.post(`/data/remove/${name}`)
                    .then(this.getList())
                    .catch(e => http.generalErrorHandler(e));
            },
            train(id) {
                let name = this.dataList[id].name;
                this.$router.push(
                    {name: 'classifier-create', params: {dataName: name}}
                );
            },
        }
    }
</script>

<style scoped>
</style>