<template>
    <div>
        <el-dialog
                :title="title"
                v-model:visible="visible"
                width="30%"
                center>
            <span>{{this.message}}</span>
            <div v-if="progress">
                <el-progress :text-inside="true" :stroke-width="26" :percentage="percentage"/>
            </div>
            <template v-slot:footer>
<span  class="dialog-footer">
                <el-button @click="visible = false">{{buttonText}}</el-button>
            </span>
</template>
        </el-dialog>
    </div>
</template>

<script>
    export default {
        name: "AsyncProgress",
        props: ['title'],
        data() {
            return {
                visible: false,
                message: "",
                progress: false,
                percentage: 0,
                buttonText: "Hide",
            }
        },
        methods: {
            show(status) {
                if (status === "Pending") {
                    this.message = "Waiting...";
                    this.progress = false;
                } else if (status.Progress !== undefined) {
                    this.progress = true;
                    this.message = status.Progress[1];
                    this.percentage = parseInt((status.Progress[0] * 100).toFixed(0));
                } else if (status.Ok !== undefined) {
                    this.progress = false;
                    this.buttonText = "Close";
                    this.message = "Finished";
                } else if (status.Error !== undefined) {
                    this.progress = false;
                    this.buttonText = "Close";
                    this.message = "Error: " + status.Error.err;
                } else {
                    this.progress = false;
                    this.message = "Invalid message from server: " + JSON.stringify(status);
                }
                this.visible = true;
            }
        }
    }
</script>

<style scoped>

</style>