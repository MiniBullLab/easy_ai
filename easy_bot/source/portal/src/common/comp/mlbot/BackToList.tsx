import React from "react";
import { useHistory } from "react-router-dom";
import { useTranslation } from "react-i18next";
import NormalButton from "common/comp/NormalButton";

const BackToList: React.FC = () => {
  const history = useHistory();
  const { t } = useTranslation();

  const backToList = () => {
    const toPath = "/task-list";
    history.push({
      pathname: toPath,
    });
  };
  return (
    <div className="back-to-list">
      <NormalButton onClick={backToList}>{t("btn.backToList")}</NormalButton>
    </div>
  );
};

export default BackToList;
