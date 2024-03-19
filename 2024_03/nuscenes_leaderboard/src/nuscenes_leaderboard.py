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
