import React, { useState, useEffect } from "react";
import { useHistory } from "react-router-dom";
import { useDispatch, useMappedState } from "redux-react-hook";
import classNames from "classnames";
import { useTranslation } from "react-i18next";
import Axios from "assets/config/http";

import Breadcrumbs from "@material-ui/core/Breadcrumbs";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import Typography from "@material-ui/core/Typography";
import MLink from "@material-ui/core/Link";

import { IState } from "store/Store";
import LeftMenu from "common/LeftMenu";
import InfoBar from "common/InfoBar";
import InfoSpan from "common/InfoSpan";

import Bottom from "common/Bottom";
import Step from "./comps/Step";
import NextButton from "common/comp/PrimaryButton";
import TextButton from "common/comp/TextButton";

import "./Creation.scss";

import { TYPE_LIST, EnumTaskType, ITypeListType } from "assets/types/index";
import {
  API_URL_NAME,
  CUR_SUPPORT_LANGS,
  URL_ML_IMAGE_TASKS,
  taskNameIsValid,
} from "assets/config/const";

// ML Comp
import MLInput from "common/comp/mlbot/MLInput";
import { ACTION_TYPES } from "store/types";

const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
});

// const dispatch = useDispatch();

const StepOne: React.FC = () => {
  const { t, i18n } = useTranslation();

  const { apiUrl } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);

  const [nameStr, setNameStr] = useState("en_name");
  const [taskName, setTaskName] = useState("");
  const [showNameRequiredError, setShowNameRequiredError] = useState(false);
  const [showNameFormatError, setShowNameFormatError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setNameStr(i18n.language + "_name");
    }
  }, [i18n.language]);

  const [taskType, setTaskType] = useState<string>(EnumTaskType.IMAGE);

  const dispatch = useDispatch();
  const updateTmpTaskInfo = React.useCallback(() => {
    dispatch({
      type: ACTION_TYPES.UPDATE_TASK_INFO,
      taskInfo: { type: taskType },
    });
  }, [dispatch, taskType]);

  // TaskType 变化时变化tmptaskinfo
  useEffect(() => {
    updateTmpTaskInfo();
  }, [taskType, updateTmpTaskInfo]);

  const history = useHistory();
  const goToHomePage = () => {
    const toPath = "/";
    history.push({
      pathname: toPath,
    });
  };
  const goToStepTwo = () => {
    // Create Task
    // if task name is empty, show tips message
    if (taskName.trim() === "") {
      setShowNameRequiredError(true);
      return;
    }
    if (!taskNameIsValid(taskName)) {
      setShowNameFormatError(true);
      return;
    }
    const taskData = {
      taskId: taskName,
      taskType: taskType,
    };
    Axios.post(API_URL + URL_ML_IMAGE_TASKS, taskData)
      .then((res) => {
        console.info(res);
        // If No next token, set is last page
        if (res.data.Status === "Success") {
          const toPath =
            "/create/step2/" + taskType.toLowerCase() + "/" + taskName;
          history.push({
            pathname: toPath,
          });
        } else {
          // setShowErrorTips(true);
          setErrorMessage(res.data.Message);
        }
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const changeDataType = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTaskType(event.target.value);
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
              <Typography color="textPrimary">
                {t("breadCrumb.create")}
              </Typography>
            </Breadcrumbs>
          </div>
          <div className="creation-content">
            <div className="creation-step">
              <Step curStep="one" />
            </div>
            <div className="creation-info">
              <div className="creation-title">
                {t("creation.step1.engineType")}
                <InfoSpan />
              </div>
              <div className="box-shadow">
                <div className="option">
                  <div className="option-title">
                    {t("creation.step1.engineOptions")}
                  </div>
                  <div className="option-list">
                    {TYPE_LIST.map((item: ITypeListType, index: number) => {
                      const optionClass = classNames({
                        "option-list-item": true,
                        "hand-point": !item.disabled,
                        active: taskType === item.value,
                      });
                      return (
                        <div key={index} className={optionClass}>
                          <label>
                            <div>
                              <input
                                disabled={item.disabled}
                                onChange={changeDataType}
                                value={item.value}
                                checked={taskType === item.value}
                                name="option-type"
                                type="radio"
                              />
                              &nbsp;{item[nameStr]}
                            </div>
                            <div className="imgs">
                              <img
                                alt={item[nameStr] as string}
                                src={item.imageSrc as string}
                              />
                            </div>
                          </label>
                        </div>
                      );
                    })}
                  </div>
                  <div className="ml-form-item">
                    <MLInput
                      inputName="taskName"
                      inputValue={taskName}
                      onChange={(
                        event: React.ChangeEvent<HTMLInputElement>
                      ) => {
                        setErrorMessage("");
                        setShowNameFormatError(false);
                        setShowNameRequiredError(false);
                        setTaskName(event.target.value);
                      }}
                      showRequiredError={showNameRequiredError}
                      requiredErrorMsg={t("creation.tips.taskNameRequired")}
                      formatErrorMsg={t("creation.tips.taskNameFormat")}
                      showFormatError={showNameFormatError}
                      optionTitle={t("creation.taskName")}
                      optionDesc={t("creation.taskNameDesc")}
                    />
                    {errorMessage && (
                      <div className="error" style={{ marginTop: "-20px" }}>
                        {errorMessage}
                      </div>
                    )}
                  </div>

                  {/* <div>
                    <ImageTips />
                  </div> */}
                </div>
              </div>
              <div className="buttons">
                <TextButton onClick={goToHomePage}>
                  {t("btn.cancel")}
                </TextButton>
                <NextButton onClick={goToStepTwo}>
                  {t("btn.createNewTask")}
                </NextButton>
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
