import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import classNames from "classnames";
import Axios from "assets/config/http";
import { useMappedState } from "redux-react-hook";
import reactStringReplace from "react-string-replace";
import randomColor from "randomcolor"; // import the script

import Breadcrumbs from "@material-ui/core/Breadcrumbs";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import Typography from "@material-ui/core/Typography";
import MLink from "@material-ui/core/Link";

// import DataLoading from "common/Loading";
import InfoSpan from "common/InfoSpan";
import InfoBar from "common/InfoBar";
import LeftMenu from "common/LeftMenu";
import Bottom from "common/Bottom";
import MLS3Input from "common/comp/mlbot/MLS3Input";
import Step from "../comps/Step";
// import NextButton from "common/comp/PrimaryButton";
import NormalButton from "common/comp/NormalButton";
import PredictLoadingButton from "common/comp/PrimaryButtonLoading";
import Loader from "react-loader-spinner";

import { IState } from "store/Store";

import {
  NER_PREDICT_TYPE,
  EnumSourceType,
  API_METHOD_LIST,
  IObjectType,
  EnumTaskType,
} from "assets/types/index";
import {
  CUR_SUPPORT_LANGS,
  API_URL_NAME,
  URL_ML_IMAGE_TASKS,
} from "assets/config/const";

import "../Creation.scss";
import PrimaryButton from "common/comp/PrimaryButton";

import MLTraining from "../comps/MLTraining";
import BackToList from "common/comp/mlbot/BackToList";
import DeployToSageMaker from "../comps/DeployToSageMaker";

// const API_URL = window.localStorage.getItem(API_URL_NAME);

const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
});

const StepOne: React.FC = (props: any) => {
  const taskIdName = props.match.params.id;

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

  // const [endPointStr, setEndpointStr] = useState("");
  const [predictType, setPredictType] = useState<string>(
    EnumSourceType.INPUT_TXT
  );

  const [showMask, setShowMask] = useState(false);
  const [modelURL, setModelURL] = useState("");

  const [curTaskStatus, setCurTaskStatus] = useState("");
  const [nerResult, setNerResult] = useState("");
  // const [nerLabelList, setNerLabelList] = useState([]);
  const [nerLabelMap, setNerLabelMap] = useState<any>({});
  const [predictTxt, setPredictTxt] = useState("");
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [showError, setShowError] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const { t, i18n } = useTranslation();

  const [titleStr, setTitleStr] = useState("en_title");
  const [descStr, setDescStr] = useState("en_desc");

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setTitleStr(i18n.language + "_title");
      setDescStr(i18n.language + "_desc");
    }
  }, [i18n.language]);

  useEffect(() => {
    // Get Model Class Data
    // setLoadingClasses(true);
    Axios.get(`${API_URL}tasks/${taskIdName}/data`)
      .then((res) => {
        // setLoadingClasses(false);
        console.info("data:", res);
        if (res && res.data) {
          setCurTaskStatus(res.data.taskStatus);
        }
      })
      .catch((err) => {
        console.error(err);
      });
  }, [API_URL, taskIdName]);

  const predictNerResult = () => {
    console.info("predictNerResult");
    const data = {
      textdata: predictTxt,
    };
    if (predictTxt.trim() === "") {
      setErrorMsg(t("creation.step2.ner.tips.textRequired"));
      setShowError(true);
      return;
    } else {
      setErrorMsg("");
      setShowError(false);
    }
    setLoadingPredict(true);
    Axios.post(`${API_URL}${URL_ML_IMAGE_TASKS}/${taskIdName}/predict`, data)
      .then((res) => {
        setLoadingPredict(false);
        if (res.data.Status === "Failed") {
          setErrorMsg(res.data.Message);
          setShowError(true);
          return;
        }
        console.info("predictNerResult=>res:", res);
        if (res.data.Results && res.data.Results.length > 0) {
          setNerResult(res.data.Results[0]);
          setErrorMsg("");
          setShowError(false);
        }
        if (res.data.Labels && res.data.Labels.length > 0) {
          const tmpLabelMap: any = {};
          res.data.Labels.forEach((element: string) => {
            tmpLabelMap[element] = {
              name: element,
              color: randomColor({
                luminosity: "light",
              }),
            };
          });
          setNerLabelMap(tmpLabelMap);
        }
      })
      .catch((err) => {
        setLoadingPredict(false);
        setErrorMsg(t("creation.tips.epNotReady"));
        setShowError(true);
        console.error(err);
      });
  };

  const changePredictType = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPredictType(event.target.value);
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
                {t("creation.step2.ner.desc")}
              </div>

              <div className="model-step-title">
                {t("creation.step2.ner.step1.name")}
              </div>

              <div className="box-shadow card-list">
                <div>
                  {/* <div className="option-title">Choose Source Data</div> */}
                  <div className="padding-20">
                    <MLS3Input
                      API_URL={apiUrl}
                      taskIdName={taskIdName}
                      optionTitle={t("creation.step2.ner.step1.selectSource")}
                      optionDesc={t(
                        "creation.step2.ner.step1.selectSourceDesc"
                      )}
                      changeSyncStatus={(status: string) => {
                        console.info("status");
                      }}
                      inputName="annotation"
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
                      taskType={EnumTaskType.NER}
                    />
                  </div>
                </div>
              </div>

              <div className="model-step-title">
                {t("creation.step2.ner.step3.name")}
              </div>

              <div className="box-shadow card-list p-relative">
                {showMask && (
                  <div className="p-cover">
                    {t("creation.step2.image.step3.ready")}
                  </div>
                )}
                <div className="option">
                  <div className="option-title">
                    {t("creation.step2.ner.step3.choseTestText")}
                  </div>
                  <div className="option-content padding-20">
                    <div>
                      <div className="upload-title">
                        {t("creation.step2.ner.step3.inputTestText")}{" "}
                        <InfoSpan />
                      </div>
                      {NER_PREDICT_TYPE.map(
                        (item: IObjectType, typIndex: number) => {
                          const stClass = classNames({
                            "no-show": true,
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
                                    disabled={
                                      item.value === EnumSourceType.S3URL
                                    }
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
                      {predictType === EnumSourceType.INPUT_TXT && (
                        <div className="input-content">
                          <div className="input-area">
                            <textarea
                              rows={10}
                              value={predictTxt}
                              onChange={(event) => {
                                setErrorMsg("");
                                setShowError(false);
                                setPredictTxt(event.target.value);
                              }}
                              placeholder={t(
                                "creation.step2.ner.step3.textareaPlaceholder"
                              )}
                              style={{ width: "100%", padding: "10px" }}
                            ></textarea>
                            <div className="error">
                              {showError && <span>{errorMsg}</span>}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="">
                              {loadingPredict ? (
                                <PredictLoadingButton disabled={true}>
                                  <Loader
                                    type="ThreeDots"
                                    color="#fff"
                                    height={10}
                                  />
                                </PredictLoadingButton>
                              ) : (
                                <PrimaryButton
                                  disabled={loadingPredict}
                                  onClick={predictNerResult}
                                >
                                  {t("btn.runTestData")}
                                </PrimaryButton>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                      {predictType === EnumSourceType.S3URL && (
                        <MLS3Input
                          API_URL={apiUrl}
                          taskIdName={taskIdName}
                          optionTitle="S3 location"
                          optionDesc={t(
                            "creation.step2.image.step1.s3BucketDesc"
                          )}
                          changeSyncStatus={(status: string) => {
                            console.info("status");
                          }}
                          inputName="s3Url"
                          placeholder="s3://bucket/path-to-your-data/"
                        />
                      )}
                    </div>
                  </div>
                  {loadingPredict ? (
                    <div className="detecting" style={{ paddingBottom: 40 }}>
                      {t("tips.detect")}
                    </div>
                  ) : (
                    <div className="ner-result">
                      <div>
                        {Object.keys(nerLabelMap).map((item) => {
                          return (
                            <span
                              className="tag-item"
                              key={item}
                              style={{
                                backgroundColor: nerLabelMap[item].color,
                              }}
                            >
                              {item}
                            </span>
                          );
                        })}
                      </div>
                      {nerResult && (
                        <div className="ner-content">
                          {reactStringReplace(
                            nerResult,
                            /\{\{(.+?)\}\}/g,
                            (match, i) => (
                              <span
                                className="ner-item"
                                key={i}
                                style={{
                                  backgroundColor:
                                    nerLabelMap[match.split(":")[0]]?.color,
                                }}
                              >
                                {match.split(":")[1]}
                              </span>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="padding-20">
                    <div className="text-right">
                      <a
                        rel="noopener noreferrer"
                        target="_blank"
                        className="no-underline"
                        href={modelURL}
                      >
                        <PrimaryButton>
                          {t("btn.downloadNERModel")}
                        </PrimaryButton>
                      </a>
                      <DeployToSageMaker taskId={taskIdName} />
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
              <div className="box-shadow card-list no-show">
                <div className="option">
                  <div className="option-title">
                    {t("creation.step2.image.step5.name")}
                    <div className="title-desc">
                      {t("creation.step2.image.step5.desc")}
                    </div>
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
                          defaultValue={defaultTxtValue}
                          rows={10}
                          style={{ width: "100%" }}
                        ></textarea>
                      </div>
                    </div>
                  </div>
                  <div className="add-another-class border-top-1px-eee">
                    <span>
                      <i>▸</i> {t("creation.step2.image.step5.exploreMore")})
                    </span>
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
