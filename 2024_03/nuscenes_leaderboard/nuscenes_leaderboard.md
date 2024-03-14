# Nuscenes 榜单处理

最近有个关于 Nuscenes 数据集打榜任务。在调研的过程中需要看 Nuscenes 榜单中的算法。但是算法太多，且都是折叠的，需要一个个点开看，很不方便。发现榜单是用 json 文件组织的。故写了一个可以轻松处理榜单 json 的类。

关于如何获取 Nuscenes json 的 URL 看前文：[获取游览器具体下载地址](../get_download_url/get_download_url.md)。

```python
import pandas as pd
import requests


class NusLeaderBoard:
    """
    这是一个可以批处理 Nuscenes 3D Objecet 榜单的类
    已完成的功能：从网页上获取榜单数据，并导出为 Excel 文件
    TODO:
    description 英文翻译中文
    根据年份、传感器模态过滤
    根据 NDS、mAP 排序
    """
    def __init__(self,
                 detection_url="https://nuscenes.org/detection.json") -> None:
        self.detection_url = detection_url
        self.json_data = self.get_detection_json()

    def get_detection_json(self):
        # 发送HTTP请求并获取JSON响应
        response = requests.get(self.detection_url)
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON数据
            json_data = response.json()
            return json_data
        else:
            print("Failed to fetch JSON data. Status code:",
                  response.status_code)

    def export2excel(self, json_data=None, output_path='output.xlsx'):
        data = []
        for method in json_data:
            method_name = method['meta']['submit_meta']["method_name"]
            method_description_en = method['meta']['submit_meta'][
                "method_description"]
            # method_description_cn = translator.translate(method_description_en, src='en', dest='zh-cn')
            data.append([method_name, method_description_en])
        df = pd.DataFrame(data, columns=['method_name', 'method_description'])
        # 将数据导出到Excel文件
        df.to_excel(output_path, index=False)


if __name__ == '__main__':
    nlb = NusLeaderBoard()
    nlb.export2excel(nlb.json_data, 'output.xlsx')

```

已完成的功能：

* 从网页上获取榜单数据
* 并导出为 Excel 文件

TODO:

* description 英文翻译中文
* 根据年份、传感器模态过滤
* 根据 NDS、mAP 排序

待续。。。

## 日期

2024/03/14：文章撰写日期
