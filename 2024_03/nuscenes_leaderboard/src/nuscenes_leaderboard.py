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
