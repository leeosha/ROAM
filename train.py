#组件名

component_name = "xdl_submitter"
component_version = "3.0.0"

#xdl任务的config.json配置，必填
config_template_name = "config.json"

# xdl任务的入口脚本，对应config.json中script配置项，必填
xdl_main_script_name = "main.py"

# 依赖的资源列表，多个资源之间用逗号分隔，可选
inputs.model_conf_list = "main.py,config.json,cal_op.py"

#自定义变量，用于渲染config.json中的变量，可选
custom_param = "-Dbizdate=${bizdate} -Dproject=trip_algo -Dsupply_table_project=trip_algo -Dsupply_table=trip_ad_roam_supply  -Ddemand_table=trip_ad_roam_demand -Doutput_odps_table_name=trip_ad_roam_output"

inputs.odps_project = "trip_algo"
