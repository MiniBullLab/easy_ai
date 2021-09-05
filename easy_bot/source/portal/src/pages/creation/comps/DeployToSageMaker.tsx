import React, { useState } from "react";
import Axios from "assets/config/http";
import { useMappedState } from "redux-react-hook";
import { useTranslation } from "react-i18next";

import Loader from "react-loader-spinner";
import PrimaryButton from "common/comp/PrimaryButton";
import DeployButtonLoading from "common/comp/PrimaryButtonLoading";

import { API_URL_NAME } from "assets/config/const";

import { IState } from "store/Store";
const mapState = (state: IState) => ({
  apiUrl: state.apiUrl,
});

interface DeployProps {
  taskId: string;
}

const DeployToSageMaker: React.FC<DeployProps> = (props) => {
  const { taskId } = props;
  const { apiUrl } = useMappedState(mapState);
  const API_URL = apiUrl || window.localStorage.getItem(API_URL_NAME);
  const [loading, setLoading] = useState(false);
  const [showError, setShowError] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const { t } = useTranslation();

  const deployToSageMaker = () => {
    setLoading(true);
    setShowError(false);
    Axios.post(`${API_URL}tasks/${taskId}/deploy`, {})
      .then((res) => {
        setLoading(false);
        console.info("res:", res);
        // Failed to show Message
        if (res.data.Message) {
          setShowError(true);
          setErrorMsg(res.data.Message);
        } else {
          // Success to redirect endpoint url
          window.open(res.data.EndpointUrl, "_blank");
        }
      })
      .catch((err) => {
        setLoading(false);
      });
  };

  return (
    <div style={{ marginLeft: 15, display: "inline-block" }}>
      {loading ? (
        <DeployButtonLoading disabled={true} style={{ minWidth: 200 }}>
          <Loader type="ThreeDots" color="#ffffff" height={10} />
        </DeployButtonLoading>
      ) : (
        <PrimaryButton
          onClick={() => {
            deployToSageMaker();
          }}
        >
          {t("btn.deployToSageMaker")}
        </PrimaryButton>
      )}

      {showError && (
        <div style={{ position: "absolute" }} className="error">
          {errorMsg}
        </div>
      )}
    </div>
  );
};

export default DeployToSageMaker;
