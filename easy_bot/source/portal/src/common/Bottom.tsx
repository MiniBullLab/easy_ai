import React, { useState } from "react";
import { useTranslation } from "react-i18next";

import "./Bottom.scss";
import LanguageIcon from "assets/images/language.png";
import FeedbackIcon from "assets/images/feedback.svg";

import { URL_FEEDBACK } from "assets/config/const";

const langList = [
  {
    id: "en",
    name: "English",
  },
  {
    id: "zh",
    name: "中文(简体)",
  },
];

const getCurrentLangObj = (id: string) => {
  let defaultItem = null;
  langList.forEach((item) => {
    if (id === item.id) {
      defaultItem = item;
    }
  });
  return defaultItem ? defaultItem : langList[0];
};

const EN_LANGUAGE_LIST = ["en", "en_US", "en_GB"];
const ZH_LANGUAGE_LIST = ["zh", "zh_CN", "zh_TW"];

const Bottom: React.FC = () => {
  const { t, i18n } = useTranslation();
  if (EN_LANGUAGE_LIST.indexOf(i18n.language) >= 0) {
    i18n.language = "en";
  }
  if (ZH_LANGUAGE_LIST.indexOf(i18n.language) >= 0) {
    i18n.language = "zh";
  }
  const initLang = getCurrentLangObj(i18n.language);
  const [currentLang, setCurrentLang] = useState(initLang);

  const changeSelectLang: any = (event: any) => {
    const newLang = JSON.parse(event.target.getAttribute("data-lang"));
    setCurrentLang(newLang);
    i18n.changeLanguage(newLang.id);
    setShowLang(false);
  };

  const [showLang, setShowLang] = useState(false);
  const toggleShowLang = () => {
    setShowLang(!showLang);
  };
  return (
    <div className="page-bottom">
      <a rel="noopener noreferrer" href={URL_FEEDBACK} target="_blank">
        <div className="item feedback">
          <img alt="feedback" src={FeedbackIcon} />
          {t("bottom.feedback")}
        </div>
      </a>
      <div className="item language">
        {showLang ? (
          <div className="language-select">
            <ul>
              {langList.map((item: any, index) => {
                return (
                  <li
                    key={index}
                    data-lang={JSON.stringify(item)}
                    onClick={changeSelectLang}
                  >
                    {item.name}
                  </li>
                );
              })}
            </ul>
          </div>
        ) : (
          ""
        )}
        <span onClick={toggleShowLang}>
          <img alt="language" src={LanguageIcon} />{" "}
          <span>{currentLang.name}</span>
        </span>
      </div>

      <span className="privacy">{t("bottom.use")}</span>
      <span className="privacy">{t("bottom.privacy")}</span>
      <span className="notice">{t("bottom.copy")}</span>
    </div>
  );
};

export default Bottom;
