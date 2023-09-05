# 天准数据格式说明

数据采集任务由天准负责，这里给出数据集的文档。

有两种类型的文件：cfg.json 文件；[timestamp].json 文件。

## cfg.json

此文件是在车上采集数据时使用。

## [timestamp].json

录的每一个包就是一个文件夹，里面存放了连续帧。文件夹以 timestamp 命名，其中一个文件就是[timestamp].json。[timestamp].json 是一个字典格式的 json。字典的最外层，包含了头部信息。头部信息的每个 key 都是从 cfg.json 获取。我们吉利需要用哪个 key 就从 cfg.json 里面拿。也就是说，[timestamp].json 的最外层 key，是 cfg.json 的子集。最终，天准给到我们的数据格式就是 [timestamp].json。下面详细介绍这个 json 文件。

### 最外层

```json
{
    "car_id": 101,
    "scene_id": "170106.json",
    "nori_path": "",
    "origin_data_path": "/media/nvidia/test/20230723_170106/",
    "parse_data_path": "./",
    "map_nori_id": "",
    "scene_tags": [],
    "weather": "",              //天气
    "fps": 10,                  //帧率
    "interval": 20,
    "calibrated_sensors": {},   //标定信息
    "frames": [],               //数据
    "prelabel_th_info": {}
}
```

### calibrated_sensors

下面的每个传感器的说明，都在 information 字段。gnss 也是如此。

```json
"calibrated_sensors": {
        "cam_back_120": {                   //120度
            "extrinsic": {},                //相机外参
            "intrinsic": {},                //相机内参
            "sensor_type": "camera",
            "T_lidar_to_pixel": []          //点云投影到图片用的矩阵
        },
        "cam_back_left_120": {},
	"cam_back_right_120": {},
	"cam_front_70_left": {},
	"cam_frontcenter_120_left": {},
	"cam_frontcenter_120_right": {},
	"cam_front_left_120": {},
	"cam_front_right_120": {},
	"front_lidar": {
	    "sensor_type": "lidar",
            "extrinsic": {},                //激光雷达外参
            "lidar_gnss": {}                //
	},
	"inno_lidar": {},                   //图达通雷达
	"s_front_lidar": {},                //速腾雷达
	"s_back_lidar": {},
	"s_left_lidar": {},
	"s_right_lidar": {},
	"lidar_ego": {},                    //
	"gnss": {},                         //
	"odom_data": {}                     //
}
```

#### cam_back_120.extrinsic

```json
"extrinsic": {
                "transform": {
                    "translation": {                 //平移变换
                        "x": -0.003564,
                        "y": 1.187667,
                        "z": -1.02016
                    },
                    "rotation": {                    //旋转变换，四元素
                        "w": 0.5053456,
                        "x": 0.4980484,
                        "y": 0.504734,
                        "z": -0.4917491
                    }
                },
                "euler_degree": {                    //欧拉角，与四元素等价
                    "RotX": 90.321916,
                    "RotY": 1.1632106,
                    "RotZ": -89.6072872
                },
                "calib_status": 1,
                "information": "cam_back_120_from_footprint_RFU",    //哪两个坐标系转换用的 RFU 后轴中心接地点，RFU -> cam
                "calib_time": "2022-10-26 21:24:32"
            },
```

#### cam_back_120.intrinsic

```json
"intrinsic": {
                "distortion_model": "fisheye",
                "K": [                       // 内参矩阵
                    [
                        1857.7024121,
                        0.0,
                        1928.58188749
                    ],
                    [
                        0.0,
                        1850.43825544,
                        1094.9183145
                    ],
                    [
                        0.0,
                        0.0,
                        1.0
                    ]
                ],
                "D": [                       //畸变参数
                    [
                        -0.02393361
                    ],
                    [
                        0.00729782
                    ],
                    [
                        -0.00326589
                    ],
                    [
                        0.00041365
                    ]
                ],
                "resolution": [
                    3840,
                    2166
                ]
            },
```

#### cam_back_120.T_lidar_to_pixel

```json
"T_lidar_to_pixel": [
                [
                    -1923.018367,
                    1863.267755,
                    26.517754,
                    -1967.081213
                ],
                [
                    -1084.18389,
                    40.763661,
                    -1856.299906,
                    1080.856874
                ],
                [
                    -0.999978,
                    0.003088,
                    -0.005727,
                    -1.020133
                ]
            ]
```

#### front_lidar.extrinsic

```json
"extrinsic": {
                "transform": {
                    "translation": {
                        "x": 0.896555,
                        "y": -0.00838634,
                        "z": 2.00893
                    },
                    "rotation": {
                        "w": 0.994705,
                        "x": 0.0046067,
                        "y": -0.0222089,
                        "z": -0.1002373
                    }
                },
                "euler_degree": {
                    "RotX": 0.0047171,
                    "RotY": -0.0451215,
                    "RotZ": -0.2007573
                },
                "calib_status": 1,
                "information": "footprint_RFU_from_front_lidar",
                "calib_time": "2022-10-26 21:24:32"
            },
```

#### front_lidar.lidar_gnss

```json
"lidar_gnss": {
                "transform": {
                    "translation": {
                        "x": -0.019096344957827187,
                        "y": -0.013858796918889263,
                        "z": -0.6318629479105103
                    },
                    "rotation": {
                        "w": 0.7009327006561435,
                        "x": -0.002099011909586967,
                        "y": 0.007799234418566003,
                        "z": -0.7131816845954276
                    }
                },
                "euler_degree": {
                    "RotX": 0.46884532364147297,
                    "RotY": 0.7980087460186799,
                    "RotZ": -90.99582574503303
                },
                "calib_status": 0,
                "information": "ins_from_footprint_RFU",
                "calib_time": "2022-10-26 21:24:32"
            }
```

#### gnss

```json
"gnss": {
            "sensor_type": "gnss",
            "gnss_ego": {
                "transform": {
                    "translation": {
                        "x": -0.009514857455192427,
                        "y": 0.08179527069452604,
                        "z": 0.3512820553962932
                    },
                    "rotation": {
                        "w": 0.9999497029242196,
                        "x": 0.007850509191857927,
                        "y": 0.005603692268140503,
                        "z": -0.0027495018016733376
                    }
                },
                "euler_degree": {
                    "RotX": 0.9014157430866968,
                    "RotY": 0.6396433579344067,
                    "RotZ": -0.3201165600777473
                },
                "calib_status": 0,
                "information": "ego_from_ins",
                "calib_time": "2022-09-20 08:27:25"
            }
        },
```

#### odom_data

```json
"odom_data": {
            "pose": {
                "pose": {
                    "position": {
                        "x": 341157.28090987384,
                        "y": 3349580.3046274465,
                        "z": 18.03206059988588
                    },
                    "orientation": {
                        "x": 0.0019291236757610246,
                        "y": -0.0033413818595033977,
                        "z": -0.19704046727901217,
                        "w": 0.9803877640523576
                    },
                    "linear_velocity": {
                        "x": 15.788263145483935,
                        "y": -6.54115971004685,
                        "z": 0.1965563991472572
                    },
                    "angular_velocity": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    }
                }
            }
        }
```

### frames

```json
            "frame_id": 1,
            "sensor_data": {
                "front_lidar": {
                    "nori_id": "JILI-03,1690102866603412961",                 //数据 id
                    "timestamp": "1690102866.603412961",                      //时间辍
                    "file_path": "lidar/front_lidar/1690102866603412961.pcd"
                },
                "fuser_lidar": {
                    "nori_id": "JILI-03,1690102866603412961",
                    "timestamp": "1690102866.603412961",
                    "file_path": "lidar/fuser_lidar/1690102866603412961.pcd"
                },
                "inno_lidar": {
                    "nori_id": "JILI-03,1690102866599861145",
                    "timestamp": "1690102866.599861145",
                    "file_path": "lidar/inno_lidar/1690102866599861145.pcd"
                },
                "radar1": {
                    "nori_id": "JILI-03,1690102866616000000",
                    "timestamp": "1690102866.616000000",
                    "file_path": "radar/radar1/1690102866616000000.pcd"
                }
            },
            "origin_frame_id": null,
            "is_key_frame": false,                             //是否关键帧
            "vehicle_report_data": null,
            "tags": [],
            "no_nan_frame_id": 0,
            "pre_labels": [],                                  //预标注的框
            "timestamp_aligned": {},
            "has_nan": true,                                   //是否有数据丢帧，所有传感器都采集到了
            "ins_data": {                                      //惯导，具体什么含义未知
                "localization": {
                    "linear_velocity": {
                        "x": -0.004761,
                        "y": 0.08606,
                        "z": -1.057617
                    },
                    "angular_velocity": {
                        "x": 0.732422,
                        "y": -0.518799,
                        "z": 4.638672
                    },
                    "orientation": {
                        "x": 0.006923,
                        "y": -0.008647,
                        "z": -0.717666,
                        "w": 0.696299
                    },
                    "position": {
                        "x": 833978.222534565,
                        "y": 0.22136551136035812,
                        "z": -0.281
                    }
                },
                "true_north_heading": -91.732866,
                "timestamp": "1690102866.608000000"
            }
```

### prelabel_th_info

```json
"prelabel_th_info": {
        "car": 0.58,
        "truck": 0.44,
        "construction_vehicle": 0.38,
        "bus": 0.46,
        "motorcycle": 0.33,
        "bicycle": 0.34,
        "tricycle": 0.42,
        "cyclist": 0.48,
        "pedestrian": 0.4
    }
```
