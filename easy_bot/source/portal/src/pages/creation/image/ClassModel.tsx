import React, { useState, useEffect, useCallback } from "react";
import { useMappedState } from "redux-react-hook";
import classNames from "classnames";
import Webcam from "react-webcam";
import Axios from "assets/config/http";
import { v4 as uuidv4 } from "uuid";
import { useTranslation } from "react-i18next";
import { useLongPress } from "use-long-press";
import { createQueue } from "best-queue";
import CloseIcon from "@material-ui/icons/Close";
import ThreeSixtyIcon from "@material-ui/icons/ThreeSixty";
import Swal from "sweetalert2";

import Loading from "common/Loading";
import PrimaryButton from "common/comp/PrimaryButton";
import TextButton from "common/comp/TextButton";
import InfoSpan from "common/InfoSpan";
import MLS3Input from "common/comp/mlbot/MLS3Input";

import { IState } from "store/Store";

import {
  IModelObj,
  ImageObjectType,
  CameraType,
  EnumTaskType,
} from "assets/types/index";

import { SOURCE_TYPE, EnumSourceType, IObjectType } from "assets/types/index";

import {
  CUR_SUPPORT_LANGS,
  API_URL_NAME,
  URL_ML_IMAGE_TASKS,
} from "assets/config/const";

type ClassModelProps = {
  taskIdName: string;
  defaultModelObj: IModelObj | null;
  openCamera: boolean;
  videoId: string;
  toggleVideo: any;
  inputChange: any;
  parentTabDisable: any;
  tabChanged: boolean;
  uploadImg: any;
};

const videoConstraints = {
  width: 224,
  height: 224,
  screenshotFormat: "image/jepg",
  screenshotQuality: 0.8,
  audio: false,
  facingMode: CameraType.FACING_MODE_USER,
};

const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
});

const ClassModel: React.FC<ClassModelProps> = (props: ClassModelProps) => {
  const {
    openCamera,
    videoId,
    taskIdName,
    defaultModelObj,
    inputChange,
    toggleVideo,
    parentTabDisable,
    // tabChanged,
    uploadImg,
  } = props;

  const { t, i18n } = useTranslation();
  const TASK_NAME = EnumTaskType.IMAGE;
  const MaxKeys = 20;

  const { apiUrl } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);
  const webcamRef = React.useRef<Webcam>(null);
  const [inputType, setInputType] = useState(EnumSourceType.S3URL);
  const [titleStr, setTitleStr] = useState("en_title");
  const [descStr, setDescStr] = useState("en_desc");
  const [longPress, setLongPress] = useState(false);
  const [curImgData, setCurImgData] = useState<ImageObjectType | null>(null);
  const [imgList, setImgList] = useState<ImageObjectType[]>([]);
  const [loadingImg, setLoadingImg] = useState(false);
  const [isAbleEdit, setIsAbleEdit] = useState(true);
  const [facingMode, setFacingMode] = useState(CameraType.FACING_MODE_USER);
  // const [curCategoryName, setCurCategoryName] = useState<string | undefined>(
  //   ""
  // );
  const [nextToken, setNextToken] = useState("");
  const [sampleCount, setSampleCount] = useState(
    defaultModelObj?.imageCount || 0
  );

  const queue = createQueue({
    max: 20,
  });

  const changeSourceType = (event: any) => {
    if (event.target.value === EnumSourceType.WEBCAMERA) {
      toggleVideo(true);
      initGetImageByCategory();
    } else {
      toggleVideo(false);
    }
    // setCurCategoryName(defaultModelObj?.modelName);
    setInputType(event.target.value);
  };

  const callback = () => {
    console.info("LONG PRESS");
    setLongPress(true);
  };

  const bind = useLongPress(callback, {
    onStart: (event: any) => {
      console.log("Press started");
    },
    onFinish: () => {
      setLongPress(false);
      queue.pause();
      queue.clear();
      console.log("Long press finished");
    },
    onCancel: () => {
      setLongPress(false);
      queue.pause();
      queue.clear();
      console.log("Press cancelled");
    },
    threshold: 300,
    captureEvent: true,
  });

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setTitleStr(i18n.language + "_title");
      setDescStr(i18n.language + "_desc");
    }
  }, [i18n.language]);

  const asyncTask = useCallback(() => {
    const jsonStr = JSON.stringify(curImgData);
    if (curImgData && curImgData.data) {
      setImgList([curImgData, ...imgList]);
      setSampleCount((count) => {
        return count + 1;
      });
      return new Promise((r) => {
        Axios.post(
          `${API_URL + URL_ML_IMAGE_TASKS}/${taskIdName}/data/${
            defaultModelObj?.modelName
          }`,
          jsonStr
        ).then((res) => {
          console.info("res:", res);
        });
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    API_URL,
    curImgData,
    defaultModelObj,
    imgList,
    longPress,
    // sampleCount,
    taskIdName,
  ]);

  const capture = useCallback(() => {
    if (!defaultModelObj || !defaultModelObj?.modelName) {
      Swal.fire(t("modelNameEmpty"));
      return false;
    }
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      const objectData = {
        job: taskIdName,
        task: TASK_NAME,
        class: defaultModelObj?.modelName,
        filename: `${defaultModelObj?.modelName}_${uuidv4()}.jpg`,
        data: imageSrc,
      };
      console.info("capture=====objectData:", objectData);
      setCurImgData(objectData);
      // Disabled Input
      setIsAbleEdit(false);
      uploadImg(true);
      // setImgList(tmpImg);
      queue.add(asyncTask, 1);
      queue.run();
    }
  }, [TASK_NAME, asyncTask, defaultModelObj, queue, t, taskIdName, uploadImg]);

  useEffect(() => {
    if (longPress) {
      const id = setInterval(() => {
        capture();
      }, 50);
      return () => clearInterval(id);
    }
  }, [capture, longPress]);

  const loadMoreImages = useCallback(() => {
    Axios.get(
      `${API_URL + URL_ML_IMAGE_TASKS}/${taskIdName}/data/${
        defaultModelObj?.modelName
      }`,
      { params: { MaxKeys: MaxKeys, ContinuationToken: nextToken } }
    ).then((res) => {
      if (res.data && res.data.URLs && res.data.URLs.length > 0) {
        const tmpImgList: ImageObjectType[] = [];
        res.data.URLs.forEach((element: string) => {
          tmpImgList.push({
            data: element,
          });
        });
        setNextToken(res.data.NextContinuationToken);
        setImgList([...imgList, ...tmpImgList]);
      }
    });
  }, [API_URL, defaultModelObj, imgList, nextToken, taskIdName]);

  const initGetImageByCategory = useCallback(() => {
    if (defaultModelObj?.modelName && !defaultModelObj?.isNew) {
      setSampleCount(defaultModelObj.imageCount);
      setNextToken("");
      setLoadingImg(true);
      parentTabDisable(true);
      setImgList([]);
      Axios.get(
        `${API_URL + URL_ML_IMAGE_TASKS}/${taskIdName}/data/${
          defaultModelObj?.modelName
        }`,
        { params: { MaxKeys: MaxKeys } }
      )
        .then((res) => {
          parentTabDisable(false);
          setLoadingImg(false);
          console.info(res);
          if (res.data.URLs && res.data.URLs.length > 0) {
            const tmpImgList: ImageObjectType[] = [];
            res.data.URLs.forEach((element: string) => {
              tmpImgList.push({
                data: element,
              });
            });
            setSampleCount(res.data.SampleCount);
            setImgList(tmpImgList);
            setNextToken(res.data.NextContinuationToken);
          }
        })
        .catch((err) => {
          parentTabDisable(false);
          setLoadingImg(false);
          console.error(err);
        });
    } else {
      setSampleCount(0);
      setNextToken("");
      setIsAbleEdit(true);
      setCurImgData(null);
      setImgList([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultModelObj]);

  // Get Image Data by Class
  useEffect(() => {
    initGetImageByCategory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultModelObj]);

  return (
    <>
      <div className="box-shadow card-list">
        <div className="option">
          <div className="option-title">
            {!defaultModelObj?.isNew && (
              <span>{defaultModelObj?.modelName}</span>
            )}
            {defaultModelObj?.isNew && (
              <input
                disabled={!isAbleEdit}
                className="option-input"
                name="modelName"
                type="text"
                onChange={(e) => {
                  // setCurCategoryName(e.target.value);
                  inputChange(e);
                }}
                value={defaultModelObj?.modelName}
                placeholder={t("creation.comp.className")}
              />
            )}
          </div>
        </div>
        <div className="model-info">
          <div className="upload-title">
            {t("creation.step2.image.step1.uploadImage")}
            <InfoSpan />
          </div>
          {SOURCE_TYPE.map((item: IObjectType, typIndex: number) => {
            const stClass = classNames({
              "st-item": true,
              active: inputType === item.value,
            });
            return (
              <div key={typIndex} className={stClass}>
                <label>
                  <div>
                    <input
                      onChange={(e) => {
                        changeSourceType(e);
                      }}
                      value={item.value}
                      checked={inputType === item.value}
                      type="radio"
                    />
                    {item[titleStr]}
                  </div>
                  <div className="desc">{item[descStr]}</div>
                </label>
              </div>
            );
          })}
          {loadingImg ? (
            <Loading />
          ) : (
            <div>
              {inputType === EnumSourceType.S3URL && (
                <MLS3Input
                  // categoryName={curCategoryName}
                  defaultModelObj={defaultModelObj}
                  taskType={TASK_NAME}
                  API_URL={apiUrl}
                  taskIdName={taskIdName}
                  optionTitle={t("creation.step2.image.step1.s3Location")}
                  optionDesc={t("creation.step2.image.step1.s3BucketDesc")}
                  inputName="s3Url"
                  placeholder="s3://bucket/path-to-your-data/"
                  changeSyncStatus={(status: string) => {
                    console.info("status:", status);
                    if (status === "start") {
                      setIsAbleEdit(false);
                    } else if (status === "success") {
                      uploadImg(true);
                    } else {
                      setIsAbleEdit(true);
                    }
                  }}
                />
              )}
              {inputType === EnumSourceType.WEBCAMERA && (
                <div className="webcam-content">
                  <div className="video">
                    {openCamera ? (
                      <div className="video-info">
                        <span
                          className="close-video"
                          onClick={() => {
                            toggleVideo(false);
                          }}
                        >
                          <CloseIcon fontSize="small" />
                        </span>
                        <span
                          className="switch-video"
                          onClick={() => {
                            setFacingMode((prevState) =>
                              prevState === CameraType.FACING_MODE_USER
                                ? CameraType.FACING_MODE_ENVIRONMENT
                                : CameraType.FACING_MODE_USER
                            );
                          }}
                        >
                          <ThreeSixtyIcon fontSize="small" />
                        </span>
                        <Webcam
                          id={videoId}
                          audio={false}
                          width={224}
                          height={224}
                          ref={webcamRef}
                          screenshotFormat="image/jpeg"
                          videoConstraints={{
                            ...videoConstraints,
                            facingMode,
                          }}
                        />
                        <div className="text-center">
                          <PrimaryButton
                            onClick={() => {
                              capture();
                            }}
                            className="full-width"
                            {...bind}
                          >
                            {t("creation.step2.image.step1.holdToRecord")}
                          </PrimaryButton>
                        </div>
                      </div>
                    ) : (
                      <div className="video-thumb">
                        <TextButton
                          className="full-width"
                          onClick={() => toggleVideo(true)}
                        >
                          {t("creation.step2.image.step1.enableVideo")}
                        </TextButton>
                      </div>
                    )}
                  </div>
                  <div className="image-list">
                    <div className="image-count">
                      {sampleCount}{" "}
                      {t("creation.step2.image.step1.imageSamples")}
                    </div>
                    <div className="image-items">
                      {imgList.map((image: ImageObjectType, index: number) => {
                        return (
                          <div key={index} className="item">
                            <img
                              width="100%"
                              height="100%"
                              alt=""
                              src={image?.data || ""}
                            />
                          </div>
                        );
                      })}
                      {nextToken && (
                        <span
                          onClick={() => {
                            if (nextToken) {
                              loadMoreImages();
                            }
                          }}
                          className="item load-more"
                        >
                          {t("loadMore")}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ClassModel;
