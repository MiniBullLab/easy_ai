import React, { useEffect, useState } from "react";
import { useDispatch } from "redux-react-hook";
import { useTranslation } from "react-i18next";
import MenuIcon from "@material-ui/icons/Menu";

import "./TopBar.scss";
import jwt_decode from "jwt-decode";

import Logo from "assets/images/logo.svg";
import MLSignOut from "./comp/SignOut";

import {
  TOKEN_STORAGE_KEY,
  ID_TOKEN_STORAGE_KEY,
  OPENID_SIGNOUT_URL,
  OPENID_SIGNIN_URL,
  AUTH_TYPE_NAME,
  OPEN_ID_TYPE,
} from "assets/config/const";
import { ACTION_TYPES } from "store/types";

const authType = localStorage.getItem(AUTH_TYPE_NAME);
const SignOutUrl = localStorage.getItem(OPENID_SIGNOUT_URL) || "/";
const SignInUrl = localStorage.getItem(OPENID_SIGNIN_URL) || "/";

const TopBar: React.FC = () => {
  const { t } = useTranslation();
  const [curUserEmail, setCurUserEmail] = useState("");

  const dispatch = useDispatch();

  const openLeftMenu = React.useCallback(() => {
    dispatch({ type: ACTION_TYPES.OPEN_SIDE_BAR });
    // localStorage.setItem("drhIsOpen", "open");
  }, [dispatch]);

  const openInNewTab = () => {
    const popWin = window.open(SignOutUrl, "_blank");
    popWin?.blur();
    window.location.href = SignInUrl;
    window.focus();
  };

  const openIdSignOut = () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    localStorage.removeItem(ID_TOKEN_STORAGE_KEY);
    openInNewTab();
  };

  useEffect(() => {
    setTimeout(() => {
      if (authType === OPEN_ID_TYPE) {
        const userIdToken = localStorage.getItem(ID_TOKEN_STORAGE_KEY) || "";
        if (userIdToken) {
          const myDecodedToken: any = jwt_decode(userIdToken);
          setCurUserEmail(myDecodedToken?.email);
        }
      } else {
        const authDataEmail = localStorage.getItem("authDataEmail");
        if (authDataEmail) {
          setCurUserEmail(authDataEmail);
        }
      }
    }, 100);
  }, []);

  return (
    <div className="drh-top-bar">
      <div className="logo">
        <span onClick={openLeftMenu} className="logo-menu">
          <MenuIcon />
        </span>
        <img
          className="logo-img"
          width="22"
          alt="Machine Learning Bot"
          src={Logo}
        />{" "}
        <span className="name">Machine Learning Bot</span>
      </div>
      <div className="options">
        {/* <div className="item">Language</div> */}
        <div className="user-item">{curUserEmail}</div>
        <div className="logout-item">
          {authType === OPEN_ID_TYPE ? (
            <div className="logout-btn-style" onClick={openIdSignOut}>
              (<span>{t("signOut")}</span>)
            </div>
          ) : (
            <MLSignOut className="logout-btn-style" />
          )}
        </div>
      </div>
      {/* <div className="options">
        <div className="user-item">
          <img alt="alert" src={AlertIcon} />
          {curUserEmail}
        </div>
        {curUserEmail && (
          <div className="logout-item">
            <div className="logout-btn-style" onClick={openIdSignOut}>
              (<span>{t("signOut")}</span>)
            </div>
          </div>
        )}
      </div> */}
    </div>
  );
};

export default TopBar;
