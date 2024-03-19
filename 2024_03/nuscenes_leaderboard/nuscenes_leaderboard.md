# Nuscenes 榜单处理

最近有个关于 Nuscenes 数据集打榜任务。在调研的过程中需要看 Nuscenes 榜单中的算法。但是算法太多，且都是折叠的，需要一个个点开看，很不方便。发现榜单是用 json 文件组织的。故写了一个可以轻松处理榜单 json 的类。

关于如何获取 Nuscenes json 的 URL 看前文：[获取游览器具体下载地址](../get_download_url/get_download_url.md)。

```python
import pandas as pd
import requests
from datetime import datetime


class NusLeaderBoard:
    """
    这是一个可以批处理 Nuscenes 3D Objecet 榜单的类
    已完成的功能：
    1.从网页上获取榜单数据，并导出为 Excel 文件
    2.根据 NDS、mAP、Date 排序
    TODO:
    description 英文翻译中文
    根据年份、传感器模态过滤
    """

    def __init__(self, detection_url) -> None:
        self.detection_url = detection_url  # "https://nuscenes.org/detection.json"
        self.json_data = self.get_detection_json()

    def get_detection_json(self):
        """
        抓取 Nuscenes 3D Object Detection 榜单数据
        """
        response = requests.get(self.detection_url)
        if response.status_code == 200:
            json_data = response.json()
            return json_data

    def _parse_datetime(self, time_str):
        """
        将时间字符串解析为 datetime 类型。时间字符串有两种类型:
            formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S']
        自动匹配传入的字符串 time_str 格式
        """
        formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S']
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                pass
        raise ValueError("Unsupported datetime format: {}".format(time_str))

    def sort(self, key="NDS", reverse=True):
        """
        根据 NDS、mAP 排序
        key: 排序关键字，"Date", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "NDS"
        """
        if key == "Date":
            self.json_data = sorted(
                self.json_data,
                key=lambda x: self._parse_datetime(x["meta"]["submit_meta"][
                    "submitted_at"]),
                reverse=reverse)
        else:
            self.json_data = sorted(
                self.json_data,
                key=lambda x: x["result"][0]["test_split"][key],
                reverse=reverse)

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
    nlb = NusLeaderBoard("https://nuscenes.org/detection.json")
    nlb.sort(key="NDS")
    nlb.export2excel(nlb.json_data, 'output.xlsx')

```

已完成的功能：

* 从网页上获取榜单数据
* 并导出为 Excel 文件
* 根据 NDS、mAP、Date 排序

TODO:

* description 英文翻译中文
* 根据年份、传感器模态过滤

待续。。。

## 日期

* 2024/03/19：根据 NDS、mAP、Date 排序
* 2024/03/14：文章撰写日期
