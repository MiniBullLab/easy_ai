import React, { useCallback, useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useDispatch } from "redux-react-hook";
import Axios from "assets/config/http";
import NormalButton from "common/comp/NormalButton";
import Swal from "sweetalert2";
import Loader from "react-loader-spinner";

import { URL_ML_IMAGE_TASKS } from "assets/config/const";
import { EnumTaskType, IModelObj } from "assets/types";

import { ACTION_TYPES } from "store/types";

type MLInputProps = {
  // categoryName?: string;
  defaultModelObj?: IModelObj | null;
  taskType?: string;
  API_URL: string;
  taskIdName: string;
  isHidden?: boolean;
  optionTitle: string;
  optionDesc: string;
  isOptional?: boolean;
  defaultValue?: string;
  // inputValue: string;
  inputName: string;
  placeholder: string;
  className?: string;
  showRequiredError?: boolean;
  requiredErrorMsg?: string;
  showFormatError?: boolean;
  formatErrorMsg?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  changeSyncStatus: any;
};

const MLInput: React.FC<MLInputProps> = (props: MLInputProps) => {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const {
    API_URL,
    taskIdName,
    optionTitle,
    optionDesc,
    isOptional,
    inputName,
    placeholder,
    className,
    showRequiredError,
    requiredErrorMsg,
    showFormatError,
    formatErrorMsg,
    taskType,
    defaultModelObj,
    // onChange,
    changeSyncStatus,
  } = props;

  const [inputValue, setInputValue] = useState("");
  const [showError, setShowError] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [successMsg, setSuccessMsg] = useState("");
  const [loadingImport, setLoadingImport] = useState(false);

  const importDataToS3 = useCallback(() => {
    if (
      taskType === EnumTaskType.IMAGE &&
      defaultModelObj?.modelName?.trim() === ""
    ) {
      Swal.fire(t("modelNameEmpty"));
      return false;
    }
    changeSyncStatus("start");
    const data = {
      OriginURIs: [inputValue],
      ClassId:
        taskType === EnumTaskType.IMAGE
          ? defaultModelObj?.modelName
          : undefined,
    };
    console.info("ADD data:", data);
    setLoadingImport(true);
    dispatch({
      type: ACTION_TYPES.SET_S3_IMPORT,
      s3IsImporting: true,
    });
    Axios.post(`${API_URL + URL_ML_IMAGE_TASKS}/${taskIdName}/s3data`, data)
      .then((res) => {
        setLoadingImport(false);
        dispatch({
          type: ACTION_TYPES.SET_S3_IMPORT,
          s3IsImporting: false,
        });
        console.info(res);
        if (res.data && res.data.Status === "Failed") {
          changeSyncStatus("error");
          setSuccessMsg("");
          setShowSuccess(false);
          setErrorMsg(res.data.Message);
          setShowError(true);
        } else {
          changeSyncStatus("success");
          setSuccessMsg(res.data.Message);
          setShowSuccess(true);
          setErrorMsg("");
          setShowError(false);
        }
      })
      .catch((err) => {
        changeSyncStatus("error");
        setLoadingImport(false);
        dispatch({
          type: ACTION_TYPES.SET_S3_IMPORT,
          s3IsImporting: false,
        });
        console.error(err);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [API_URL, defaultModelObj, inputValue, t, taskIdName, taskType]);

  // Monitor Object Change
  useEffect(() => {
    console.info("S3 INPUT defaultModelObj:", defaultModelObj);
    if (defaultModelObj?.isNew) {
      setInputValue("");
      setSuccessMsg("");
      setShowSuccess(false);
      setErrorMsg("");
      setShowError(false);
    }
  }, [defaultModelObj]);

  useEffect(() => {
    // Get Model Class Data
    setLoadingImport(true);
    // Hide Messages
    setSuccessMsg("");
    setShowSuccess(false);
    setErrorMsg("");
    setShowError(false);
    let mounted = true;
    if (taskIdName) {
      Axios.get(`${API_URL}tasks/${taskIdName}/s3data`)
        .then((res) => {
          if (mounted) {
            setLoadingImport(false);
            console.info("data:", res);
            if (res && res.data) {
              if (defaultModelObj && defaultModelObj.modelName) {
                setInputValue(
                  res.data?.OriginURIs?.[defaultModelObj.modelName] || ""
                );
              } else {
                setInputValue(res.data?.OriginURIs[0] || "");
              }
            }
          }
        })
        .catch((err) => {
          setLoadingImport(false);
          console.error(err);
        });
    }
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskIdName]);

  return (
    <>
      <div className="s3-content">
        <div className="s3-title">
          {optionTitle}
          {isOptional && (
            <span>
              - <i>{t("optional")}</i>
            </span>
          )}
        </div>
        <div className="s3-desc">{optionDesc}</div>
        <div className="select-wrap">
          <div className="s3-input">
            <div className="input">
              <input
                // defaultValue={defaultValue}
                value={inputValue}
                name={inputName}
                onChange={(event) => {
                  setInputValue(event.target.value);
                  setErrorMsg("");
                  setShowError(false);
                  setSuccessMsg("");
                  setShowSuccess(false);
                }}
                className={className}
                placeholder={placeholder}
                type="text"
              />
            </div>
            <div className="button">
              {loadingImport ? (
                <NormalButton disabled={true}>
                  <Loader type="ThreeDots" color="#888" height={10} />
                </NormalButton>
              ) : (
                <NormalButton onClick={importDataToS3}>
                  {t("btn.importData")}
                </NormalButton>
              )}
            </div>
          </div>
          <div className="error">
            {showError && <span>{errorMsg}</span>}
            {showSuccess && (
              <span style={{ color: "#1d8309" }}>{successMsg}</span>
            )}
            {showRequiredError && <span>{requiredErrorMsg}</span>}
            {showFormatError && <span>{formatErrorMsg}</span>}
          </div>
        </div>
      </div>
      {/* <div className="title">
        {optionTitle}
        {isOptional && (
          <span>
            - <i>{t("optional")}</i>
          </span>
        )}
      </div>
      <div className="desc">{optionDesc}</div>
      <div>
        
      </div> */}
    </>
  );
};

export default MLInput;
