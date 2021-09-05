import React, { useEffect, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import classNames from "classnames";
import Webcam from "react-webcam";
import Axios from "assets/config/http";
import { useMappedState } from "redux-react-hook";
import { DropzoneArea } from "material-ui-dropzone";
import CloseIcon from "@material-ui/icons/Close";
import ThreeSixtyIcon from "@material-ui/icons/ThreeSixty";

import Breadcrumbs from "@material-ui/core/Breadcrumbs";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import Typography from "@material-ui/core/Typography";
import MLink from "@material-ui/core/Link";

import Swal from "sweetalert2";

import DataLoading from "common/Loading";
import InfoSpan from "common/InfoSpan";
import InfoBar from "common/InfoBar";
import LeftMenu from "common/LeftMenu";
import Bottom from "common/Bottom";
import Step from "../comps/Step";
import TextButton from "common/comp/TextButton";
import PredictLoadingButton from "common/comp/PrimaryButtonLoading";
import Loader from "react-loader-spinner";

import { IState } from "store/Store";

import {
  PREDICT_TYPE,
  EnumSourceType,
  // TRAINING_STATUS_LIST,
  IObjectType,
  CameraType,
  IModelObj,
  EnumTaskType,
} from "assets/types/index";
import {
  CUR_SUPPORT_LANGS,
  API_URL_NAME,
  URL_ML_IMAGE_TASKS,
} from "assets/config/const";

import "../Creation.scss";
import PrimaryButton from "common/comp/PrimaryButton";
import ProgressBar from "common/comp/ProgressBar";
import InfoIcon from "assets/images/info.svg";
import ClassModel from "./ClassModel";
import MLTraining from "../comps/MLTraining";
import BackToList from "common/comp/mlbot/BackToList";

import DeployToSageMaker from "../comps/DeployToSageMaker";

const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
  s3IsImporting: state.s3IsImporting,
});

const videoConstraints = {
  width: 224,
  height: 224,
  screenshotFormat: "image/jepg",
  screenshotQuality: 0.8,
  audio: false,
  facingMode: CameraType.FACING_MODE_USER,
};

const MemoizedClassModel = React.memo(ClassModel);

const StepOne: React.FC = (props: any) => {
  const taskIdName = props.match.params.id;
  const webcamRef = useRef<Webcam>(null);

  const { apiUrl, s3IsImporting } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);

  const predictVideo = {
    openCamera: true,
    videoId: "predict",
    inputType: "",
  };

  const [predictImgData, setPredictImgData] = useState<string | null>("");
  const [predictType, setPredictType] = useState("");
  const [showPredictVideo, setShowPredictVideo] = useState(false);
  const [resultList, setResultList] = useState([]);

  const [showMask, setShowMask] = useState(true);
  const [modelURL, setModelURL] = useState("");

  const [loadingClasses, setLoadingClasses] = useState(false);
  const [modelList, setModelList] = useState<IModelObj[]>([]);
  const [curModelObj, setCurModelObj] = useState<IModelObj | null>(null);
  const [enableClassVideo, setEnableClassVideo] = useState(true);
  const [curTaskStatus, setCurTaskStatus] = useState("");
  const [disableChangeTab, setDisableChangeTab] = useState(false);
  const [tabChanged, setTabChanged] = useState(true);
  const [facingMode, setFacingMode] = useState(CameraType.FACING_MODE_USER);
  const [predictErrorMsg, setPredictErrorMsg] = useState("");
  const [predictLoading, setPredictLoading] = useState(false);

  const { t, i18n } = useTranslation();

  const [titleStr, setTitleStr] = useState("en_title");
  const [descStr, setDescStr] = useState("en_desc");

  const setInitOneModel = () => {
    const initObj: IModelObj = {
      modelName: "class_1",
      imageCount: 0,
      isNew: true,
      index: 0,
    };
    setModelList([initObj]);
    setCurModelObj(initObj);
  };

  // Monitor S3 Is Importing
  useEffect(() => {
    console.info("s3IsImporting:", s3IsImporting);
  }, [s3IsImporting]);

  useEffect(() => {
    // Get Model Class Data
    setLoadingClasses(true);
    Axios.get(`${API_URL}tasks/${taskIdName}/data`)
      .then((res) => {
        setLoadingClasses(false);
        console.info("data:", res);
        if (res && res.data) {
          setCurTaskStatus(res.data.taskStatus);
          if (
            res.data.taskData &&
            res.data.taskData.Classes &&
            res.data.taskData.Classes.length > 0
          ) {
            const tmpList: IModelObj[] = [];
            res.data.taskData.Classes.forEach(
              (element: string, index: number) => {
                tmpList.push({
                  modelName: element,
                  imageCount: res.data.taskData.SampleCount[index],
                  index: index,
                  isNew: false,
                });
              }
            );
            setModelList(tmpList);
            setCurModelObj(tmpList[0]);
          } else {
            setInitOneModel();
          }
        } else {
          setInitOneModel();
        }
      })
      .catch((err) => {
        setLoadingClasses(false);
        console.error(err);
      });
  }, [API_URL, taskIdName]);

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setTitleStr(i18n.language + "_title");
      setDescStr(i18n.language + "_desc");
    }
  }, [i18n.language]);

  useEffect(() => {
    if (curTaskStatus === "Hosting" || curTaskStatus === "Completed") {
      setShowMask(false);
    } else {
      setShowMask(true);
    }
  }, [curTaskStatus]);

  const addAnotherClass = () => {
    console.info("add another class");
    const tmpList = [...modelList];
    const tmpListNameArr: string[] = [];
    tmpList.forEach((element) => {
      tmpListNameArr.push(element.modelName);
    });
    let newClassName = "class_" + (modelList.length + 1);
    if (tmpListNameArr.indexOf(newClassName) >= 0) {
      newClassName = newClassName + "1";
    }
    const newModelObj: IModelObj = {
      index: modelList.length,
      modelName: newClassName,
      imageCount: 0,
      isNew: true,
    };
    tmpList.push(newModelObj);
    setCurModelObj(newModelObj);
    setModelList(tmpList);
  };

  const predictResult = () => {
    console.info("predictResult");
    const imageSrc = webcamRef?.current?.getScreenshot() || "";
    setPredictImgData(imageSrc);
    const imageDataParam = {
      imagedata: imageSrc,
    };
    const imageDataParamStr = JSON.stringify(imageDataParam);
    // setPredictLoading(true);
    setPredictErrorMsg("");
    setPredictLoading(true);
    Axios.post(
      `${API_URL}${URL_ML_IMAGE_TASKS}/${taskIdName}/predict`,
      imageDataParamStr
    )
      .then((res) => {
        setPredictLoading(false);
        console.info("res:", res);
        if (res && res.data && res.data.Message) {
          setPredictErrorMsg(res.data.Message);
        } else {
          setResultList(res?.data?.Results || []);
        }
      })
      .catch((err) => {
        setPredictLoading(false);
        // setEndpointNotReady(true);
        console.error(err);
      });
  };

  const changePredictType = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPredictImgData("");
    setResultList([]);
    if (event.target.value === EnumSourceType.WEBCAMERA) {
      setShowPredictVideo(true);
      setEnableClassVideo(false);
    } else {
      setEnableClassVideo(false);
    }
    setPredictType(event.target.value);
  };

  const turnOnPredictVideo = () => {
    setShowPredictVideo(true);
    setEnableClassVideo(false);
  };

  const changeModelTab = (element: IModelObj) => {
    if (disableChangeTab) {
      return;
    }
    setTabChanged(true);
    setCurModelObj(element);
  };

  const convertBase64 = (file: any) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);

      fileReader.onload = () => {
        resolve(fileReader.result);
      };

      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  };

  const predictUploadImage = async (files: any) => {
    console.info("files:", files);
    if (files && files.length > 0) {
      const imageBase64 = await convertBase64(files[0]);
      setPredictImgData(imageBase64 as string);
      const imageDataParam = {
        // endpoint: endpointName,
        imagedata: imageBase64,
      };
      const imageDataParamStr = JSON.stringify(imageDataParam);
      // setPredictLoading(true);
      setPredictErrorMsg("");
      setPredictLoading(true);
      Axios.post(
        `${API_URL}${URL_ML_IMAGE_TASKS}/${taskIdName}/predict`,
        imageDataParamStr
      )
        .then((res) => {
          setPredictLoading(false);
          if (res && res.data && res.data.Message) {
            setPredictErrorMsg(res.data.Message);
          } else {
            setResultList(res?.data?.Results || []);
          }
        })
        .catch((err) => {
          console.error(err);
          // setEndpointNotReady(true);
          setPredictLoading(false);
        });
    }
  };

  return (
    <div className="drh-page">
      <LeftMenu />
      <div className="right">
        <InfoBar />
        <div className="padding-left-40">
          <div className="page-breadcrumb">
            <Breadcrumbs
              separator={<NavigateNextIcon fontSize="small" />}
              aria-label="breadcrumb"
            >
              <MLink color="inherit" href="/#/">
                {t("breadCrumb.home")}
              </MLink>
              <MLink color="inherit" href="/#/task-list">
                {t("breadCrumb.tasks")}
              </MLink>
              <Typography color="textPrimary">
                {/* {t("breadCrumb.create")} */}
                {taskIdName}
              </Typography>
            </Breadcrumbs>
          </div>
          <div className="creation-content">
            <div className="creation-step">
              <Step curStep="two" />
            </div>
            <div className="creation-info">
              <BackToList />
              <div className="creation-title">{taskIdName}</div>
              <div className="creation-desc">
                {t("creation.step2.image.desc")}
              </div>
              <div className="model-step-title">
                {t("creation.step2.image.step1.name")}
              </div>
              {loadingClasses ? (
                <div>
                  <DataLoading />
                </div>
              ) : (
                <div>
                  <div className="model-class-list">
                    <div className="model-class-tabs">
                      {modelList.map((element: IModelObj, index: number) => {
                        return (
                          <span
                            onClick={() => {
                              changeModelTab(element);
                            }}
                            key={index}
                            className={classNames({
                              "class-tab": true,
                              disabled: disableChangeTab,
                              active:
                                element.modelName === curModelObj?.modelName,
                            })}
                          >
                            {element.modelName}
                          </span>
                        );
                      })}
                      <span
                        className={classNames({
                          "class-tab": true,
                          disabled: s3IsImporting,
                        })}
                        onClick={() => {
                          if (!s3IsImporting) {
                            addAnotherClass();
                          }
                        }}
                      >
                        +
                      </span>
                    </div>
                  </div>
                  <div>
                    <MemoizedClassModel
                      inputChange={(
                        event: React.ChangeEvent<HTMLInputElement>
                      ) => {
                        if (curModelObj) {
                          const preTmpModelList = modelList;
                          preTmpModelList[curModelObj.index].modelName =
                            event.target.value;
                          setModelList(preTmpModelList);
                          setCurModelObj({
                            index: curModelObj.index,
                            modelName: event.target.value,
                            imageCount: curModelObj.imageCount,
                            isNew: true,
                          });
                        }
                      }}
                      toggleVideo={(value: boolean) => {
                        setEnableClassVideo(value);
                        if (value) {
                          setShowPredictVideo(false);
                        }
                      }}
                      parentTabDisable={(disable: boolean) => {
                        console.info("LOADING:", disable);
                        setTabChanged(false);
                        setDisableChangeTab(disable);
                      }}
                      uploadImg={(upload: boolean) => {
                        console.info("uploadImg:", upload);
                        if (upload) {
                          // setCurModelObj({ ...curModelObj, isNew: false });
                          if (curModelObj) {
                            console.info("curModelObj:", curModelObj);
                            const tmpModelList = modelList;
                            tmpModelList[curModelObj.index]["isNew"] = false;
                            setModelList(tmpModelList);
                            setTabChanged(false);
                          }
                        }
                      }}
                      tabChanged={tabChanged}
                      defaultModelObj={curModelObj}
                      taskIdName={taskIdName}
                      openCamera={enableClassVideo}
                      videoId="video1"
                    />
                  </div>
                </div>
              )}

              <div className="model-step-title">
                {t("creation.step2.image.step2.name")}
              </div>

              <div className="box-shadow card-list">
                <div className="option">
                  <div className="option-content padding-20">
                    <MLTraining
                      firstClassNew={
                        modelList?.length > 0 ? modelList?.[0]?.isNew : true
                      }
                      secondClassNew={
                        modelList?.length > 1 ? modelList?.[1]?.isNew : true
                      }
                      taskType={EnumTaskType.IMAGE}
                      classCount={modelList.length}
                      trainingComplete={(
                        complete: boolean,
                        modelUrl: string
                      ) => {
                        setModelURL(modelUrl);
                        if (complete) {
                          setShowMask(false);
                        } else {
                          setShowMask(true);
                        }
                      }}
                      curTaskStatus={curTaskStatus}
                      taskIdName={taskIdName}
                    />
                  </div>
                </div>
              </div>

              <div className="model-step-title">
                {t("creation.step2.image.step3.name")} <InfoSpan />
              </div>

              <div className="box-shadow card-list p-relative">
                {showMask && (
                  <div className="p-cover">
                    {t("creation.step2.image.step3.ready")}
                  </div>
                )}
                <div className="option">
                  <div className="option-title">
                    {t("creation.step2.image.step3.chooseTestImage")}
                  </div>
                  <div className="option-content padding-20">
                    <div>
                      <div className="upload-title">
                        {t("creation.step2.image.step3.upload")} <InfoSpan />
                      </div>
                      {PREDICT_TYPE.map(
                        (item: IObjectType, typIndex: number) => {
                          const stClass = classNames({
                            "st-item": true,
                            active: predictType === item.value,
                          });
                          return (
                            <div key={typIndex} className={stClass}>
                              <label>
                                <div>
                                  <input
                                    onChange={(e) => {
                                      changePredictType(e);
                                    }}
                                    // defaultValue={formDefaultValue.sourceType}
                                    value={item.value}
                                    checked={predictType === item.value}
                                    name="predictImageType"
                                    type="radio"
                                  />
                                  {item[titleStr]}
                                </div>
                                <div className="desc">{item[descStr]}</div>
                              </label>
                            </div>
                          );
                        }
                      )}

                      <div className="text-center">
                        <div className="object-predict">
                          <div className="item drop-zone">
                            {predictType === EnumSourceType.WEBCAMERA && (
                              <div>
                                {showPredictVideo ? (
                                  <div className="video-info">
                                    <span
                                      className="close-video"
                                      onClick={() => {
                                        setShowPredictVideo(false);
                                      }}
                                    >
                                      <CloseIcon fontSize="small" />
                                    </span>
                                    <span
                                      className="switch-video"
                                      onClick={() => {
                                        setFacingMode((prevState) =>
                                          prevState ===
                                          CameraType.FACING_MODE_USER
                                            ? CameraType.FACING_MODE_ENVIRONMENT
                                            : CameraType.FACING_MODE_USER
                                        );
                                      }}
                                    >
                                      <ThreeSixtyIcon fontSize="small" />
                                    </span>
                                    <div className="web-cam">
                                      <Webcam
                                        id={predictVideo.videoId}
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
                                    </div>
                                    <div className="text-center">
                                      {predictLoading ? (
                                        <PredictLoadingButton
                                          className="full-width"
                                          disabled={true}
                                        >
                                          <Loader
                                            type="ThreeDots"
                                            color="#fff"
                                            height={10}
                                          />
                                        </PredictLoadingButton>
                                      ) : (
                                        <PrimaryButton
                                          className="full-width"
                                          onClick={() => predictResult()}
                                        >
                                          {t("btn.clickPredict")}
                                        </PrimaryButton>
                                      )}
                                    </div>
                                  </div>
                                ) : (
                                  <div className="video-thumb">
                                    <TextButton
                                      onClick={turnOnPredictVideo}
                                      className="full-width"
                                    >
                                      {t("btn.enableVideo")}
                                    </TextButton>
                                  </div>
                                )}
                              </div>
                            )}

                            {predictType === EnumSourceType.UPLOAD && (
                              <div>
                                <DropzoneArea
                                  acceptedFiles={["image/*"]}
                                  maxFileSize={20000000}
                                  dropzoneText={t("btn.uploadFile")}
                                  showAlerts={false}
                                  showPreviews={false}
                                  showPreviewsInDropzone={false}
                                  filesLimit={1}
                                  onChange={(files) => {
                                    if (files && files[0]?.size > 2097152) {
                                      Swal.fire(
                                        t("tips.title"),
                                        t("tips.maxFileSize")
                                      );
                                    } else {
                                      predictUploadImage(files);
                                    }
                                  }}
                                />
                              </div>
                            )}
                          </div>
                          {predictErrorMsg && (
                            <div className="preview-image">
                              <div className="detecting">
                                <div className="not-ready">
                                  {predictErrorMsg}
                                </div>
                              </div>
                            </div>
                          )}
                          {!predictErrorMsg && (
                            <div className="preview-image">
                              {predictLoading ? (
                                <div className="detecting">
                                  {t("tips.detect")}
                                </div>
                              ) : (
                                predictImgData && (
                                  <div>
                                    <img alt="" src={predictImgData} />
                                  </div>
                                )
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="predict-result-list">
                      {resultList.map((element: any, index: number) => {
                        return (
                          <div key={index} className="result-item">
                            <div className="predict-icon">
                              <div>
                                <img width="20" alt="" src={InfoIcon} />
                              </div>
                            </div>
                            <div className="predict-res">
                              <div className="result-name">{element.Class}</div>
                              <div className="progress-wrap">
                                <div className="bar">
                                  <ProgressBar
                                    value={
                                      parseFloat(element.Probability) * 100
                                    }
                                  />
                                </div>
                                <div className="value">
                                  {(
                                    parseFloat(element.Probability) * 100
                                  ).toFixed(2)}
                                  %
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="text-right">
                      <div className="download-button">
                        <a
                          rel="noopener noreferrer"
                          target="_blank"
                          className="no-underline"
                          href={modelURL}
                        >
                          <PrimaryButton>
                            {t("btn.downloadModel")}
                          </PrimaryButton>
                        </a>
                        <DeployToSageMaker taskId={taskIdName} />
                      </div>
                      <div className="download-tips">
                        {/* {t("creation.step2.image.step3.chooseS3")} */}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="bottom">
          <Bottom />
        </div>
      </div>
    </div>
  );
};

export default StepOne;
