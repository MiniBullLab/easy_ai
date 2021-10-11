import React, { useState, useEffect } from "react";
import { useHistory } from "react-router-dom";
import { useMappedState } from "redux-react-hook";
import classNames from "classnames";
import { useTranslation } from "react-i18next";

import LeftMenu from "common/LeftMenu";
import Bottom from "common/Bottom";
import Card from "./comps/Card";
import NextButton from "common/comp/PrimaryButton";

import "./Home.scss";

import { IState } from "store/Store";
import { IObjectType, IHomeContentType } from "assets/types/index";

import WORK_IMG from "assets/images/works.jpg";

import {
  TOP_TITLE_INFO,
  HOW_IT_WORKS,
  BENIFITS_AND_FEATURES,
  GET_START_LINKS,
  // PRICE_LIST,
  RESOURCE_LINKS,
} from "assets/config/content";

import { CUR_SUPPORT_LANGS } from "assets/config/const";

const mapState = (state: IState) => ({
  isOpen: state.isOpen,
  lastUpdated: state.lastUpdated,
});

const Home: React.FC = () => {
  const { t, i18n } = useTranslation();
  const [titleStr, setTitleStr] = useState("en_title");
  const [subTitleStr, setSubTitleStr] = useState("en_subTitle");
  const [descStr, setDescStr] = useState("en_desc");
  const [contentStr, setContentStr] = useState("en_content");
  // const [priceStr, setPriceStr] = useState("en_price");
  // const [nameStr, setNameStr] = useState("en_name");

  useEffect(() => {
    if (CUR_SUPPORT_LANGS.indexOf(i18n.language) >= 0) {
      setTitleStr(i18n.language + "_title");
      setSubTitleStr(i18n.language + "_subTitle");
      setDescStr(i18n.language + "_desc");
      setContentStr(i18n.language + "_content");
      // setPriceStr(i18n.language + "_price");
      // setNameStr(i18n.language + "_name");
    }
  }, [i18n.language]);

  const topTitleInfo: IObjectType = TOP_TITLE_INFO;
  const howItWorks: IObjectType = HOW_IT_WORKS;
  const benifitsAndFeatures: IHomeContentType = BENIFITS_AND_FEATURES;
  const getStartLinks: IHomeContentType = GET_START_LINKS;
  const resourceLinks: IHomeContentType = RESOURCE_LINKS;
  // const priceList: IHomeContentType = PRICE_LIST;

  const { isOpen } = useMappedState(mapState);

  const history = useHistory();
  const startToCreate = () => {
    const toPath = "/create/step1";
    history.push({
      pathname: toPath,
    });
  };

  const topShowClass = classNames({
    "top-show": true,
    opened: isOpen,
  });

  const contentClass = classNames({
    "content-area": true,
    opened: isOpen,
  });

  return (
    <div className="drh-page">
      <LeftMenu />
      <div className="right">
        <div className="padding-left-40">
          <div className={topShowClass}>
            <div className="intro">
              <div className="ml">{t("top.ml")}</div>
              <div className="big">{topTitleInfo[titleStr]}</div>
              <div className="medium">{topTitleInfo[subTitleStr]}</div>
              <div className="small">{topTitleInfo[descStr]}</div>
            </div>
          </div>
          <div className={contentClass}>
            <div className="left-info">
              <div className="title">{howItWorks[titleStr]}</div>
              <div className="video">
                <div className="img-shadow">
                  <img width="100%" src={WORK_IMG} alt="" />
                </div>
              </div>

              <div className="benifit">
                <div className="title">{benifitsAndFeatures[titleStr]}</div>
                <div className="benifit-list box-info">
                  {benifitsAndFeatures.list.map(
                    (element: IObjectType, index: number) => {
                      return (
                        <div key={index} className="items">
                          <div className="name">{element[titleStr]}</div>
                          <div className="content">{element[contentStr]}</div>
                        </div>
                      );
                    }
                  )}
                </div>
              </div>
            </div>
            <div className="right-card">
              <div className="home-card start-item">
                <div className="title">{t("home.title.createTitle")}</div>
                <div className="next-button">
                  <NextButton
                    onClick={startToCreate}
                    variant="contained"
                    color="primary"
                    disableRipple
                  >
                    {t("btn.getStart")}
                  </NextButton>
                </div>
              </div>
              {/* <div className="home-card info-item">
                <div className="card">
                  <div className="title">{priceList[titleStr]}</div>
                  <div className="item-list">
                    <ul>
                      {priceList.list.map(
                        (element: IObjectType, index: number) => {
                          return (
                            <li className="price-item" key={index}>
                              <div className="label">{element[nameStr]}</div>
                              <div className="value">{element[priceStr]}</div>
                            </li>
                          );
                        }
                      )}
                    </ul>
                    <div className="link outside">
                      <a href="/#/">{t("home.priceCalc")}</a>
                    </div>
                  </div>
                </div>
              </div> */}
              <div className="home-card info-item">
                <Card contentInfo={getStartLinks} />
              </div>
              <div className="home-card info-item">
                <Card contentInfo={resourceLinks} />
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

export default Home;
