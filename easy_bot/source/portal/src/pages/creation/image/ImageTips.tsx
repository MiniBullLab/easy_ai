import React from "react";
import { useTranslation } from "react-i18next";

const StepOneS3Tips: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="tips">
      <div className="tips-title">{t("creation.step1.tips.title")}</div>
      <div className="tips-desc">{t("creation.step1.tips.desc")}</div>
      <div className="tips-list">
        <ul>
          <li>
            <span>•</span>
            {t("creation.step1.tips.line1")}
          </li>
          <li>
            <span>•</span>
            {t("creation.step1.tips.line2")}
          </li>
          <li>
            <span>•</span>
            {t("creation.step1.tips.line3")}
          </li>
        </ul>
      </div>
    </div>
  );
};

export default StepOneS3Tips;
