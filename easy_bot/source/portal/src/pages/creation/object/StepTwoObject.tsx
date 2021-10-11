import React, { useEffect, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import classNames from "classnames";
import Webcam from "react-webcam";
import Axios from "assets/config/http";
import CloseIcon from "@material-ui/icons/Close";
import ThreeSixtyIcon from "@material-ui/icons/ThreeSixty";
import { useMappedState } from "redux-react-hook";
import { DropzoneArea } from "material-ui-dropzone";

import Breadcrumbs from "@material-ui/core/Breadcrumbs";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import Typography from "@material-ui/core/Typography";
import MLink from "@material-ui/core/Link";

import Swal from "sweetalert2";

// import DataLoading from "common/Loading";
import InfoSpan from "common/InfoSpan";
import InfoBar from "common/InfoBar";
import LeftMenu from "common/LeftMenu";
import Bottom from "common/Bottom";
import MLS3Input from "common/comp/mlbot/MLS3Input";
import Step from "../comps/Step";
// import NextButton from "common/comp/PrimaryButton";
import NormalButton from "common/comp/NormalButton";
import TextButton from "common/comp/TextButton";
import { IState } from "store/Store";

import MLTraining from "../comps/MLTraining";
import BackToList from "common/comp/mlbot/BackToList";
import PredictLoadingButton from "common/comp/PrimaryButtonLoading";
import Loader from "react-loader-spinner";

import {
  PREDICT_TYPE,
  API_METHOD_LIST,
  EnumSourceType,
  IObjectType,
  CameraType,
} from "assets/types/index";
import {
  CUR_SUPPORT_LANGS,
  API_URL_NAME,
  URL_ML_IMAGE_TASKS,
} from "assets/config/const";

import "../Creation.scss";
import PrimaryButton from "common/comp/PrimaryButton";
import DeployToSageMaker from "../comps/DeployToSageMaker";

const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
});

const RESULT_THREASHOLD = 0.5;
const RESULT_IMAGE_WIDTH = 480;

// Hook
function useWindowSize() {
  // Initialize state with undefined width/height so server and client renders match
  // Learn more here: https://joshwcomeau.com/react/the-perils-of-rehydration/
  const [windowSize, setWindowSize] = useState({
    width: 0,
    height: 0,
  });

  useEffect(() => {
    // Handler to call on window resize
    function handleResize() {
      // Set window width/height to state
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }

    // Add event listener
    window.addEventListener("resize", handleResize);

    // Call handler right away so state gets updated with initial window size
    handleResize();

    // Remove event listener on cleanup
    return () => window.removeEventListener("resize", handleResize);
  }, []); // Empty array ensures that effect is only run on mount

  return windowSize;
}

const StepOne: React.FC = (props: any) => {
  const taskIdName = props.match.params.id;
  const webcamRef = useRef<Webcam>(null);
  const size = useWindowSize();

  const { apiUrl } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);

  const defaultTxtValue = JSON.stringify(
    {
      inputFilePath: "s3:src/abcdsf-erwer-fbh",
      outputFilePath: "s3:des/gbclss-frwer-bdf",
    },
    null,
    2
  );

  const predictVideo = {
    openCamera: true,
    videoId: "predict",
    inputType: "",
  };

  const [predictType, setPredictType] = useState("");
  const [showPredictVideo, setShowPredictVideo] = useState(false);

  const [showMask, setShowMask] = useState(true);
  const [modelURL, setModelURL] = useState("");

  const [curTaskStatus, setCurTaskStatus] = useState("");
  const [uploadPredictImgData, setUploadPredictImgData] = useState("");
  const [uploadImgResultList, setUploadImgResultList] = useState([]);
  const [predictLoading, setPredictLoading] = useState(false);
  const [resultImageWidth, setResultImageWidth] = useState(RESULT_IMAGE_WIDTH);
  const [predictErrorMsg, setPredictErrorMsg] = useState("");

  const { t, i18n } = useTranslation();

  const [titleStr, setTitleStr] = useState("en_title");
  const [descStr, setDescStr] = useState("en_desc");

  const videoConstraints = {
    width: 500,
    height: 375,
    screenshotFormat: "image/jepg",
    screenshotQuality: 0.8,
    audio: false,
    facingMode: CameraType.FACING_MODE_USER,
  };

  useEffect(() => {
    console.info("size:", size);
    // setHistoryHeight(size.height - 150);
    // setModalHeight(size.height - 150);
    setResultImageWidth(
      size.width < RESULT_IMAGE_WIDTH ? size.width - 100 : RESULT_IMAGE_WIDTH
    );
  }, [size]);

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setTitleStr(i18n.language + "_title");
      setDescStr(i18n.language + "_desc");
    }
  }, [i18n.language]);

  const [facingMode, setFacingMode] = useState(CameraType.FACING_MODE_USER);

  useEffect(() => {
    // Get Model Class Data
    // setLoadingClasses(true);
    Axios.get(`${API_URL}tasks/${taskIdName}/data`)
      .then((res) => {
        // setLoadingClasses(false);
        console.info("data:", res);
        if (res && res.data) {
          setCurTaskStatus(res.data.taskStatus);
        } else {
          // setInitOneModel();
        }
      })
      .catch((err) => {
        // setLoadingClasses(false);
        console.error(err);
      });
  }, [API_URL, taskIdName]);

  const predictVideoResult = () => {
    console.info("predictVideoResult");
    const imageSrc = webcamRef?.current?.getScreenshot() || "";
    // setPredictImgData(imageSrc);
    setUploadPredictImgData(imageSrc);
    const imageDataParam = {
      // endpoint: endpointName,
      imagedata: imageSrc,
    };
    const imageDataParamStr = JSON.stringify(imageDataParam);
    setPredictLoading(true);
    setPredictErrorMsg("");
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
          buildResData(res.data);
        }
      })
      .catch((err) => {
        console.error(err);
        setPredictLoading(false);
        // setEndpointNotReady(true);
      });
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

  const buildResData = (data: any) => {
    if (data && data.Result) {
      const resultObj = data.Result;
      const imgWidth = data.width;
      const imgHeight = data.height;
      if (
        resultObj.bounding_boxs &&
        resultObj.scores &&
        resultObj.bounding_boxs.length > 0 &&
        resultObj.scores.length > 0 &&
        resultObj.bounding_boxs.length === resultObj.scores.length
      ) {
        const ratio =
          imgWidth >= imgHeight
            ? resultImageWidth / imgWidth
            : imgWidth / resultImageWidth;
        const tmpResList: any = [];
        resultObj.scores.forEach((element: any, index: number) => {
          if (element > RESULT_THREASHOLD) {
            console.info(
              "resultObj.class_IDs[index]:",
              resultObj.class_IDs[index]
            );
            tmpResList.push({
              value: element,
              label: resultObj.classes[resultObj.class_IDs[index]],
              left:
                imgWidth >= imgHeight
                  ? resultObj.bounding_boxs[index][0] * ratio
                  : resultObj.bounding_boxs[index][0] / ratio,
              top:
                imgWidth >= imgHeight
                  ? resultObj.bounding_boxs[index][1] * ratio
                  : resultObj.bounding_boxs[index][1] / ratio,
              width:
                (resultObj.bounding_boxs[index][2] -
                  resultObj.bounding_boxs[index][0]) *
                ratio,
              height:
                (resultObj.bounding_boxs[index][3] -
                  resultObj.bounding_boxs[index][1]) *
                ratio,
            });
          }
        });
        console.info("tmpResList:", tmpResList);
        setUploadImgResultList(tmpResList);
      }
    } else {
      // setResultList([]);
    }
  };

  const predictUploadImage = async (files: any) => {
    console.info("files:", files);
    if (files && files.length > 0) {
      const imageBase64 = await convertBase64(files[0]);
      setUploadPredictImgData(imageBase64 as string);
      const imageDataParam = {
        // endpoint: endpointName,
        imagedata: imageBase64,
      };
      const imageDataParamStr = JSON.stringify(imageDataParam);
      setPredictLoading(true);
      setPredictErrorMsg("");
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
            buildResData(res.data);
          }
        })
        .catch((err) => {
          console.error(err);
          setPredictLoading(false);
          // setEndpointNotReady(true);
        });
    }
  };

  const changePredictType = (event: React.ChangeEvent<HTMLInputElement>) => {
    setUploadImgResultList([]);
    setUploadPredictImgData("");
    if (event.target.value === EnumSourceType.WEBCAMERA) {
      setShowPredictVideo(true);
    } else {
    }
    setPredictType(event.target.value);
  };

  const turnOnPredictVideo = () => {
    setShowPredictVideo(true);
  };

  const changeMethodType = () => {
    console.info("changeMethodType");
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
              <Typography color="textPrimary">{taskIdName}</Typography>
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
                {t("creation.step2.object.desc")}
              </div>

              <div className="model-step-title">
                {t("creation.step2.object.step1.importData")}
              </div>

              <div className="box-shadow card-list">
                <div>
                  <div className="option-title">
                    {t("creation.step2.object.step1.chooseSourceData")}
                  </div>
                  <div className="padding-lr-20">
                    <MLS3Input
                      API_URL={apiUrl}
                      taskIdName={taskIdName}
                      optionTitle={t("creation.step2.object.step1.s3Location")}
                      optionDesc={t(
                        "creation.step2.object.step1.s3LocationDesc"
                      )}
                      changeSyncStatus={(status: string) => {
                        console.info("status");
                      }}
                      inputName="s3Url"
                      placeholder="s3://bucket/path-to-your-data/"
                    />
                  </div>
                </div>
              </div>

              <div className="model-step-title">
                {t("creation.step2.image.step2.name")}
              </div>

              <div className="box-shadow card-list">
                <div className="option">
                  <div className="option-content padding-20">
                    <MLTraining
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
                {t("creation.step2.image.step3.name")}
              </div>

              <div className="box-shadow card-list p-relative">
                {showMask && (
                  <div className="p-cover">
                    {t("creation.step2.image.step3.ready")}
                  </div>
                )}
                <div className="option">
                  <div className="option-title">
                    {t("creation.step2.object.step3.chooseTestImage")}
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
                                  <div className="video-info object">
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
                                        width={250}
                                        height={200}
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
                                          onClick={() => predictVideoResult()}
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
                                  dropzoneText={t("btn.uploadFile")}
                                  acceptedFiles={["image/*"]}
                                  maxFileSize={20000000}
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
                            <div className="upload-predict-result">
                              <div className="detecting">
                                <div className="not-ready">
                                  {predictErrorMsg}
                                </div>
                              </div>
                            </div>
                          )}
                          {!predictErrorMsg && (
                            <div className="upload-predict-result">
                              {predictLoading ? (
                                <div className="detecting">
                                  {t("tips.detect")}
                                </div>
                              ) : (
                                <div
                                  style={{ width: `${resultImageWidth}px` }}
                                  className="upload-result-wrap"
                                >
                                  {uploadImgResultList.map(
                                    (element: any, index) => {
                                      return (
                                        <div
                                          key={index}
                                          className="upload-result-item"
                                          style={{
                                            left: `${element.left}px`,
                                            top: `${element.top}px`,
                                            width: `${element.width}px`,
                                            height: `${element.height}px`,
                                          }}
                                        >
                                          <span className="result-score">
                                            {element.label}:
                                            {element.value.toFixed(3)}
                                          </span>
                                        </div>
                                      );
                                    }
                                  )}
                                  {uploadPredictImgData && (
                                    <img
                                      width="100%"
                                      alt="result"
                                      src={uploadPredictImgData}
                                    />
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
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
              <div className="box-shadow card-list no-show">
                <div className="option">
                  <div className="option-title">
                    {t("creation.step2.image.step4.name")}
                    <div className="title-desc">
                      {t("creation.step2.image.step4.desc")}
                    </div>
                  </div>
                  <div className="option-content padding-20">
                    <div className="option-tranning">
                      <div className="button">
                        <NormalButton>{t("btn.launchStack")}</NormalButton>
                      </div>
                      <div className="times">
                        {t("creation.step2.image.step4.expectMin")}
                      </div>
                    </div>
                  </div>
                  <div className="add-another-class border-top-1px-eee">
                    <span>
                      <i>▸</i> {t("creation.step2.image.advancedSetting")}
                    </span>
                  </div>
                </div>
              </div>

              <div className="model-step-title no-show">
                Step 4 (Optional): Make API Call Info
              </div>

              <div className="box-shadow card-list no-show">
                <div className="option">
                  <div className="option-title">
                    Make API Call
                    {/* {t("creation.step2.image.step5.name")} */}
                    {/* <div className="title-desc">
                      {t("creation.step2.image.step5.desc")}
                    </div> */}
                  </div>
                  <div className="option-content padding-20">
                    <div className="option-tranning">
                      <div className="api-types">
                        <div className="api-title">
                          {t("creation.step2.image.step5.restApi")}
                        </div>
                        <div className="api-list">
                          {API_METHOD_LIST.map(
                            (element: IObjectType, index: number) => {
                              return (
                                <div key={index} className="api-item">
                                  <label>
                                    <input
                                      onChange={() => {
                                        changeMethodType();
                                      }}
                                      name="apiType"
                                      value={element.value}
                                      type="radio"
                                    />
                                    {element.name}
                                  </label>
                                </div>
                              );
                            }
                          )}
                        </div>
                      </div>
                      <div className="api-output">
                        <div className="endpoint">
                          {t("creation.step2.image.step5.endPoint")}:
                          https://aws-solutions-public-endpoint/ml-bot/image-classification
                          <div className="info-span">
                            <InfoSpan />
                          </div>
                        </div>
                        <div className="point-desc">
                          {t("creation.step2.image.step5.publicEndpoint")}
                        </div>
                        <textarea
                          placeholder={defaultTxtValue}
                          rows={10}
                          style={{ width: "100%" }}
                        ></textarea>
                        <div className="text-right padding-tb-20">
                          <PrimaryButton>Call API</PrimaryButton>
                        </div>
                      </div>
                    </div>
                  </div>
                  {/* <div className="add-another-class border-top-1px-eee">
                    <span>
                      <i>▸</i> {t("creation.step2.image.step5.exploreMore")})
                    </span>
                  </div> */}
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
