import React, { useState, useRef, useCallback, useEffect } from "react";
import Axios from "assets/config/http";
import Swal from "sweetalert2";
import ArrowRightSharpIcon from "@material-ui/icons/ArrowRightSharp";
import ArrowDropDownSharpIcon from "@material-ui/icons/ArrowDropDownSharp";
import { IState } from "store/Store";
import { useMappedState } from "redux-react-hook";
import { useTranslation } from "react-i18next";

import {
  TRAINING_STATUS_LIST,
  IObjectType,
  EnumTaskType,
} from "assets/types/index";
import {
  URL_ML_IMAGE_TASKS,
  API_URL_NAME,
  CUR_SUPPORT_LANGS,
} from "assets/config/const";

import NormalButton from "common/comp/NormalButton";

const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
});

const STATUS_TRANING = "Training";
const EPOCHS_DEFAULT = "80";
const IMAGE_SIZE_DEFAULT = "256";
const LANGUAGE_DEFAULT = "zh";

type MLTrainingPros = {
  firstClassNew?: boolean;
  secondClassNew?: boolean;
  taskType?: string;
  classCount?: number;
  taskIdName: string;
  curTaskStatus: string;
  trainingComplete: any;
};

const toGrayList = [
  {
    zh_name: "是",
    en_name: "Yes",
    value: "1",
  },
  {
    zh_name: "否",
    en_name: "No",
    value: "0",
  },
];

const MLTraining: React.FC<MLTrainingPros> = (props: MLTrainingPros) => {
  const {
    // firstClassNew,
    // secondClassNew,
    taskIdName,
    curTaskStatus,
    trainingComplete,
    taskType,
    classCount,
  } = props;
  const { t, i18n } = useTranslation();

  const [trainLoading, setTrainLoading] = useState(false);
  const [resStatusList, setResStatusList] = useState([]);

  const [errorMsg, setErrorMsg] = useState("");
  const [showError, setShowError] = useState(false);

  const [modelIsOk, setModelIsOk] = useState(false);
  const [modelUrl, setModelUrl] = useState("");

  const [statusNameStr, setStatusNameStr] = useState("en_Status");
  const [statusDescStr, setStatusDescStr] = useState("en_StatusMessage");

  const [showTrainSteps, setShowTrainSteps] = useState(false);

  const [cloudWatchUrl, setCloudWatchUrl] = useState("");

  const [epochs, setEpochs] = useState(EPOCHS_DEFAULT);
  const [toGray, setToGray] = useState("0");
  const [languageParam, setLanguageParam] = useState(LANGUAGE_DEFAULT);
  const [imageSize, setImageSize] = useState(IMAGE_SIZE_DEFAULT);
  const [advancedShow, setAdvancedShow] = useState(false);

  const { apiUrl } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);

  const intervalObj = useRef(0);

  const getTrainingStatus = useCallback(() => {
    Axios.get(`${API_URL + URL_ML_IMAGE_TASKS}/${taskIdName}/status`).then(
      (res) => {
        if (res.data) {
          const restTrainJobStatus = res.data.TrainingJobStatus || [];
          // const resEndpointStatus = res.data.EndpointStatus;
          const modelArtifacts = res.data?.ModelArtifacts?.[0] || "";
          console.info("===restTrainJobStatus===:", restTrainJobStatus);
          setResStatusList(restTrainJobStatus);
          setCloudWatchUrl(res.data.CloudWatchLogsUrl);
          const epochsValue =
            res.data?.HyperParameters?.EPOCHS || EPOCHS_DEFAULT;
          setEpochs(epochsValue);
          const imageSizeValue =
            res.data?.HyperParameters?.IMAGE_SIZE || IMAGE_SIZE_DEFAULT;
          setImageSize(imageSizeValue);
          const toGrayValue = res.data?.HyperParameters?.TO_GRAY || "0";
          setToGray(toGrayValue);
          const languageValue =
            res.data?.HyperParameters?.LANGUAGE || LANGUAGE_DEFAULT;
          setLanguageParam(languageValue);
          if (restTrainJobStatus.indexOf("Failed") >= 0) {
            clearInterval(intervalObj.current);
            setTrainLoading(false);
          }
          if (restTrainJobStatus.length >= 5) {
            setTrainLoading(false);
            setModelIsOk(true);
            setModelUrl(modelArtifacts);
            clearInterval(intervalObj.current);
            setModelIsOk(true);
          }
        }
      }
    );
  }, [API_URL, taskIdName]);

  const startStatusInterval = () => {
    intervalObj.current = setInterval(() => {
      getTrainingStatus();
    }, 5000) as any;
  };

  const startToTranning = () => {
    if (taskType === EnumTaskType.IMAGE && classCount && classCount < 2) {
      // Swal.fire("At least two classes");
      Swal.fire({
        title: t("creation.step2.image.twoClassesTips"),
        icon: "warning",
      });
      return;
    }
    // if (taskType === EnumTaskType.IMAGE && (firstClassNew || secondClassNew)) {
    //   // Swal.fire("At least two classes");
    //   Swal.fire({
    //     title: t("creation.step2.image.noSamplesTips"),
    //     icon: "warning",
    //   });
    //   return;
    // }

    let trainParamData: any = {
      HyperParameters: { EPOCHS: epochs ? epochs : EPOCHS_DEFAULT },
    };
    if (taskType === EnumTaskType.IMAGE) {
      if (
        Number.parseInt(imageSize) < 256 ||
        Number.parseInt(imageSize) > 1024
      ) {
        Swal.fire({
          title: t("imageSizeErrorTips"),
          icon: "warning",
        });
        return;
      }
      trainParamData = {
        HyperParameters: {
          EPOCHS: epochs ? epochs : EPOCHS_DEFAULT,
          IMAGE_SIZE: imageSize ? imageSize : IMAGE_SIZE_DEFAULT,
          TO_GRAY: toGray ? toGray : "0",
        },
      };
    }
    if (taskType === EnumTaskType.NER) {
      trainParamData = {
        HyperParameters: {
          EPOCHS: epochs ? epochs : EPOCHS_DEFAULT,
          LANGUAGE: languageParam ? languageParam : LANGUAGE_DEFAULT,
        },
      };
    }
    console.info("startToTranning");
    setModelIsOk(false);
    setTrainLoading(true);
    setShowError(false);
    setErrorMsg("");
    setCloudWatchUrl("");
    Axios.post(
      `${API_URL}${URL_ML_IMAGE_TASKS}/${taskIdName}/train`,
      trainParamData
    )
      .then((res) => {
        console.info("startToTranning=>res:", res);
        if (res.data.Status === "Failed") {
          setTrainLoading(false);
          setErrorMsg(res.data.Message);
          setShowError(true);
        } else {
          setShowTrainSteps(true);
          setErrorMsg("");
          setShowError(false);
          startStatusInterval();
        }
      })
      .catch((err) => {
        console.error(err);
        setTrainLoading(false);
      });
  };

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setStatusNameStr(i18n.language + "_Status");
      setStatusDescStr(i18n.language + "_StatusMessage");
    }
  }, [i18n.language]);

  useEffect(() => {
    if (curTaskStatus === STATUS_TRANING) {
      intervalObj.current = setInterval(() => {
        getTrainingStatus();
      }, 5000) as any;
    }
  }, [curTaskStatus, getTrainingStatus]);

  useEffect(() => {
    console.info("AAAcurTaskStatus:", curTaskStatus);
    if (curTaskStatus === STATUS_TRANING || curTaskStatus === "") {
      setTrainLoading(true);
    } else {
      setTrainLoading(false);
    }
  }, [curTaskStatus]);

  useEffect(() => {
    getTrainingStatus();
  }, [getTrainingStatus]);

  useEffect(() => {
    if (modelIsOk) {
      trainingComplete(true, modelUrl);
    } else {
      trainingComplete(false, "");
    }
  }, [modelIsOk, modelUrl, trainingComplete]);

  return (
    <>
      <div className="option-tranning">
        <div className="button">
          <NormalButton disabled={trainLoading} onClick={startToTranning}>
            {t("btn.startTraining")}
          </NormalButton>
          <span className="cloud-watch">
            {cloudWatchUrl && (
              <a rel="noopener noreferrer" href={cloudWatchUrl} target="_blank">
                {t("cloudWatch")}
              </a>
            )}
          </span>
          <div className="tips error">
            {showError && <span>{errorMsg}</span>}
          </div>
        </div>
        <div className="times">
          {(showTrainSteps || curTaskStatus !== "NotStarted") && (
            <div className="inner">
              {TRAINING_STATUS_LIST.map(
                (status: IObjectType, index: number) => {
                  let curClassName = "normal";
                  if (index === resStatusList.length - 1) {
                    curClassName = "normal";
                  }
                  if (index < resStatusList.length) {
                    curClassName = "success";
                  }
                  if (index > resStatusList.length) {
                    curClassName = "gray";
                  }
                  if (resStatusList[index] === "Failed") {
                    curClassName = "error";
                  }
                  return (
                    <div className={curClassName} key={index}>
                      {curClassName === "error"
                        ? "Failed"
                        : status[statusNameStr]}{" "}
                      <span className="desc">({status[statusDescStr]})</span>
                    </div>
                  );
                }
              )}
            </div>
          )}
        </div>
      </div>
      <div className="add-another-class border-top-1px-eee">
        <span>
          <i>
            {!advancedShow && (
              <ArrowRightSharpIcon
                onClick={() => {
                  setAdvancedShow(true);
                }}
                className="option-profession-icon"
                fontSize="large"
              />
            )}
            {advancedShow && (
              <ArrowDropDownSharpIcon
                onClick={() => {
                  setAdvancedShow(false);
                }}
                className="option-profession-icon"
                fontSize="large"
              />
            )}
          </i>{" "}
          {t("creation.step2.image.advancedSetting")}
        </span>
        {advancedShow && (
          <div className="advance-option">
            <div className="title">Epochs</div>
            <div className="input">
              <input
                disabled={trainLoading}
                className="option-input"
                value={epochs}
                type="number"
                min={1}
                onWheel={(event) => event.currentTarget.blur()}
                onChange={(event) => {
                  setEpochs(event.target.value);
                }}
                style={{ width: 150 }}
              />
            </div>
            {taskType === EnumTaskType.NER && (
              <div>
                <div className="title mt-10">{t("languageName")}</div>
                <div className="input">
                  <select
                    className="option-input"
                    style={{ width: 150 }}
                    value={languageParam}
                    onChange={(event) => {
                      console.info("event:", event.target.value);
                      setLanguageParam(event.target.value);
                    }}
                  >
                    <option value="zh">中文</option>
                    <option value="en">English</option>
                  </select>
                </div>
              </div>
            )}

            {taskType === EnumTaskType.IMAGE && (
              <div>
                <div className="title mt-10">
                  {t("imageSizeName")}
                  <span className="title-tips">{t("imageTips")}</span>
                </div>
                <div className="input">
                  <input
                    disabled={trainLoading}
                    className="option-input"
                    value={imageSize}
                    type="number"
                    min={1}
                    onWheel={(event) => event.currentTarget.blur()}
                    onChange={(event) => {
                      setImageSize(event.target.value);
                    }}
                    style={{ width: 150 }}
                  />
                </div>
                <div className="title mt-10">
                  {t("toGrayName")}
                  {/* <span className="title-tips">{t("imageTips")}</span> */}
                </div>
                <div className="input">
                  {toGrayList.map((element: any, index: number) => {
                    return (
                      <label key={index} className="radio-input">
                        <input
                          disabled={trainLoading}
                          checked={element.value === toGray ? true : false}
                          type="radio"
                          value={element.value}
                          name="toGrayRadio"
                          onChange={(event) => {
                            setToGray(event.target.value);
                          }}
                        />{" "}
                        {element[i18n.language + "_name"]}
                      </label>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
};

export default MLTraining;
