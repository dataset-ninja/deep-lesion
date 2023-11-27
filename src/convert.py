# https://www.kaggle.com/datasets/kmader/nih-deeplesion-subset

import os, csv, ast
import supervisely as sly
from collections import defaultdict
from supervisely.io.fs import get_file_name_with_ext, get_file_name, file_exists
from dotenv import load_dotenv

import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
import shutil

from tqdm import tqdm


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "NIH DeepLesion"
    dataset_path = "/home/grokhi/rawdata/deep-lesion/Images_png"
    batch_size = 30
    anns_path = "/home/grokhi/rawdata/deep-lesion/kag/DL_info.csv"

    def create_ann(image_name):
        labels = []
        tags = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        shape = name_to_shape[image_name]
        img_height = shape[0]
        img_wight = shape[1]

        ann_data = name_to_data[image_name]
        age_value = ann_data[0][-1]
        if age_value != "NaN":
            age = sly.Tag(age_meta, value=int(age_value))
            tags.append(age)
        gender_value = ann_data[0][-2]
        gender = sly.Tag(gender_meta, value=gender_value)
        tags.append(gender)
        for curr_ann_data in ann_data:
            lesion_type_value = int(ann_data[0][2])
            lesion_type = sly.Tag(lesion_type_meta, value=lesion_type_value)
            lesion_diameter_value = ann_data[0][1]
            lesion_diameter = sly.Tag(lesion_diameter_meta, value=lesion_diameter_value)
            bboxes = ast.literal_eval(str(curr_ann_data[0]))
            left = int(bboxes[0])
            top = int(bboxes[1])
            right = int(bboxes[2])
            bottom = int(bboxes[3])
            rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
            label = sly.Label(rect, obj_class, tags=[lesion_type, lesion_diameter])
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class = sly.ObjClass("lesion", sly.Rectangle)
    age_meta = sly.TagMeta("age", sly.TagValueType.ANY_NUMBER)
    gender_meta = sly.TagMeta("gender", sly.TagValueType.ONEOF_STRING, possible_values=["F", "M"])
    lesion_type_meta = sly.TagMeta("lesion type", sly.TagValueType.ANY_NUMBER)
    lesion_diameter_meta = sly.TagMeta("lesion diameter", sly.TagValueType.ANY_STRING)
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class],
        tag_metas=[age_meta, gender_meta, lesion_type_meta, lesion_diameter_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    name_to_data = defaultdict(list)
    name_to_shape = {}
    train_images_names = []
    val_images_names = []
    test_images_names = []
    with open(anns_path, "r") as file:
        csvreader = csv.reader(file)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            name_to_data[row[0]].append([row[6], row[7], row[9], row[15], row[16]])
            name_to_shape[row[0]] = ast.literal_eval(str(row[-5]))
            if row[-1] == "1":
                train_images_names.append(row[0])
            elif row[-1] == "2":
                val_images_names.append(row[0])
            else:
                test_images_names.append(row[0])

    ds_name_to_data = {
        "train": set(train_images_names),
        "val": set(val_images_names),
        "test": set(test_images_names),
    }

    for ds_name, images_names in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(list(images_names), batch_size=batch_size):
            img_pathes_batch = []
            real_images_names_batch = []
            for image_name in images_names_batch:
                im_path = os.path.join(dataset_path, image_name[:-8] + "/" + image_name[-8 + 1 :])
                if file_exists(im_path):
                    img_pathes_batch.append(im_path)
                    real_images_names_batch.append(image_name)

            img_infos = api.image.upload_paths(
                dataset.id, real_images_names_batch, img_pathes_batch
            )
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(im_name) for im_name in real_images_names_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))
    return project
