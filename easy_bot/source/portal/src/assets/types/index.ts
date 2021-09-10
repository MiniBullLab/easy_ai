// Task Type Icons
import ICON_IMAGE from "../images/option-image.png";
import ICON_OBJECT from "../images/option-object.png";
import ICON_NER from "../images/option-ner.png";

export enum EnumClassficationStatus {
  INIT = "INIT",
  UPLOADING = "UPLOADING",
  TRAINING = "TRAINING",
  TRAINING_COMPLETE = "TRAINING_COMPLETE",
}

export type ImageObjectType = {
  job?: string;
  task?: string;
  class?: string;
  filename?: string;
  data: string | null;
};
export interface IModelObj {
  index: number;
  modelName: string;
  imageCount: number;
  // imgList: ImageObjectType[];
  isNew: boolean;
}

export interface IObjectType {
  [key: string]: number | string;
}

export interface IHomeContentType {
  [key: string]: number | string | IObjectType[];
  list: IObjectType[];
}

export const TASK_STATUS_MAP: any = {
  NotStarted: {
    en_name: "Not Started",
    zh_name: "Not Started",
    class: "gray",
  },
  Training: { en_name: "Training", zh_name: "Training", class: "gray" },
  Hosting: { en_name: "Hosting", zh_name: "Hosting", class: "success" },
  Completed: {
    en_name: "Completed",
    zh_name: "Completed",
    class: "success",
  },
  Failed: { en_name: "Failed", zh_name: "Failed", class: "error" },
};

export const TRAINING_STATUS_LIST: IObjectType[] = [
  {
    name: "Starting",
    en_Status: "Starting",
    zh_Status: "启动中",
    en_StatusMessage: "Preparing the instances for training",
    zh_StatusMessage: "准备实例以进行训练",
  },
  {
    name: "Downloading",
    en_Status: "Downloading",
    zh_Status: "下载文件中",
    en_StatusMessage: "Downloading input data",
    zh_StatusMessage: "下载输入数据",
  },
  {
    name: "Training",
    en_Status: "Training",
    zh_Status: "训练中",
    en_StatusMessage:
      "Training image download completed. Training in progress.",
    zh_StatusMessage: "训练图像下载完成。训练正在进行中。",
  },
  {
    name: "Uploading",
    en_Status: "Uploading",
    zh_Status: "上传模型",
    en_StatusMessage: "Uploading generated training model",
    zh_StatusMessage: "上传生成的模型",
  },
  {
    name: "Completed",
    en_Status: "Completed",
    zh_Status: "模型完成",
    en_StatusMessage: "Training job completed",
    zh_StatusMessage: "模型训练任务完成",
  },
  // {
  //   name: "In-Service",
  //   en_Status: "In-Service",
  //   "zh_Status": "服务已就绪",
  //   en_StatusMessage: "Predict service is ready",
  //   "zh_StatusMessage": "预测服务已就绪",
  // },
];

// Task Tyep Enum
export enum EnumTaskType {
  IMAGE = "IMAGE_CLASSIFICATION",
  OBJECT = "OBJECT_DETECTION",
  TEXT = "TEXT",
  NER = "NAMED_ENTITY_RECOGNITION",
  SKELETON = "SKELETON",
  FACE = "FACE",
  CAR = "CAR",
}

export const API_METHOD_LIST: IObjectType[] = [
  {
    name: "POST",
    value: "POST",
  },
  // {
  //   name: "GET",
  //   value: "GET",
  // },
  // {
  //   name: "PUT",
  //   value: "PUT",
  // },
];

export interface ITypeListType {
  id: number | string;
  [key: string]: string | number | boolean;
  value: string;
  disabled: boolean;
}

// Task List
export const TYPE_LIST: ITypeListType[] = [
  {
    id: "1",
    en_name: "Image Classfication",
    zh_name: "图像分类",
    value: EnumTaskType.IMAGE,
    imageSrc: ICON_IMAGE,
    disabled: false,
  },
  {
    id: "2",
    en_name: "Object Detection",
    zh_name: "目标检测",
    value: EnumTaskType.OBJECT,
    imageSrc: ICON_OBJECT,
    disabled: false,
  },
  // {
  //   id: 3,
  //   en_name: "Text Classfication",
  //   "zh_name": "文本分类",
  //   value: EnumTaskType.TEXT,
  //   imageSrc: ICON_TEXT,
  //   disabled: false,
  // },
  {
    id: "4",
    en_name: "Named Entity Recognition",
    zh_name: "命名实体识别",
    value: EnumTaskType.NER,
    imageSrc: ICON_NER,
    disabled: false,
  },
  // {
  //   id: 2,
  //   en_name: "Skeleton Detection",
  //   "zh_name": "骨架识别",
  //   value: EnumTaskType.SKELETON,
  //   imageSrc: ICON_SKELETON,
  //   disabled: false,
  // },
  // {
  //   id: 3,
  //   en_name: "Face Detection",
  //   "zh_name": "面部识别",
  //   value: EnumTaskType.FACE,
  //   imageSrc: ICON_FACE,
  //   disabled: false,
  // },
  // {
  //   id: 4,
  //   en_name: "Car License Detection",
  //   "zh_name": "车牌识别",
  //   value: EnumTaskType.CAR,
  //   imageSrc: ICON_CAR,
  //   disabled: false,
  // },
];

// Camera Type
export enum CameraType {
  FACING_MODE_USER = "user",
  FACING_MODE_ENVIRONMENT = "environment",
}

// Task Tyep Enum
export enum EnumSourceType {
  WEBCAMERA = "WEBCAMERA",
  S3URL = "S3URL",
  UPLOAD = "UPLOAD",
  INPUT_TXT = "INPUT_TXT",
}

// Task List
export const SOURCE_TYPE: IObjectType[] = [
  {
    id: "1",
    en_title: "Web camera ",
    zh_title: "网络摄像头",
    value: EnumSourceType.WEBCAMERA,
    en_desc:
      "Deliver all types of content (including streaming). This is the most common choice.",
    zh_desc: "交付所有类型的内容（包括流式传输）。这是最常见的选择",
  },
  {
    id: "2",
    en_title: "Import from S3",
    zh_title: "从S3导入",
    value: EnumSourceType.S3URL,
    en_desc: "Select S3 folder and ingest images from the S3 bucket.",
    zh_desc: "选择 S3 文件夹并从 S3 存储桶中获取图像。",
  },
];

export const PREDICT_TYPE = [
  {
    id: "1",
    en_title: "Web camera ",
    zh_title: "网络摄像头",
    value: EnumSourceType.WEBCAMERA,
    en_desc:
      "Deliver all types of content (including streaming). This is the most common choice.",
    zh_desc: "交付所有类型的内容（包括流式传输）。这是最常见的选择",
  },
  {
    id: "2",
    en_title: "Upload Image",
    zh_title: "上传文件",
    value: EnumSourceType.UPLOAD,
    en_desc:
      "Upload images from local filepath, the images will be upload via brower",
    zh_desc: "从本地文件路径上传图片，图片将通过浏览器上传。",
  },
];

export const NER_PREDICT_TYPE = [
  {
    id: "1",
    en_title: "Input text",
    zh_title: "输入文本",
    value: EnumSourceType.INPUT_TXT,
    en_desc: "Enter text below for testing.",
    zh_desc: "在下面输入框中输入文本来进行测试",
  },
  {
    id: "2",
    en_title: "Import from S3",
    zh_title: "从S3导入",
    value: EnumSourceType.S3URL,
    en_desc: " Upload TXT file from S3 for testing. ",
    zh_desc: "从S3文件夹导入测试文本。",
  },
];
