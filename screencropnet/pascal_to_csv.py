#!/usr/bin/env python

# SOURCE: https://gist.github.com/rotemtam/88d9a4efae243fc77ed4a0f9917c8f6c

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path: str):
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            bbx = member.find("bndbox")
            xmin = int(bbx.find("xmin").text)
            ymin = int(bbx.find("ymin").text)
            xmax = int(bbx.find("xmax").text)
            ymax = int(bbx.find("ymax").text)
            label = member.find("name").text

            value = (
                root.find("filename").text,
                xmin,
                ymin,
                xmax,
                ymax,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                label,
            )
            xml_list.append(value)
    column_name = [
        "img_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "width",
        "height",
        "label",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # datasets = ["train", "dev", "test"]
    datasets = ["pascal_temp"]
    for ds in datasets:
        image_path = os.path.join("/Users/malcolm/Downloads", ds, "annotations")
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv("labels2_{}.csv".format(ds), index=None)
        print("Successfully converted xml to csv.")


main()
