// URL to be done
export const URL = "";
export const URL_FEEDBACK = "";
export const URL_YOUTUBE = "";
export const CUR_SUPPORT_LANGS: string[] = ["zh", "en"];
export const TOKEN_STORAGE_KEY = "__ML_BOT_AUTH_TOKEN__";
export const ID_TOKEN_STORAGE_KEY = "__ML_BOT_AUTH_TOKEN__";
export const OPENID_SIGNOUT_URL = "__ML_OPENID_SIGNOUT_URL__";
export const OPENID_SIGNIN_URL = "__ML_OPENID_SIGNIN_URL__";

export const API_URL_NAME = "ML_BOT_API_URL";
export const URL_ML_IMAGE_TASKS = "tasks";
export const OPEN_ID_TYPE = "OPENID";
export const AUTH_TYPE_NAME = "__ML_BOT_AUTH_TYPE__";

type ObjectType = {
  [key: string]: string;
};

export const converListToMap = (list: any): any => {
  const tmpMap: Record<string, unknown> = {};
  list.forEach((element: ObjectType) => {
    tmpMap[element.value] = element;
  });
  return tmpMap;
};

export const taskNameIsValid = (taskName: string): boolean => {
  // return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const REG1 = taskName && /^[A-Za-z\d-]*$/.test(taskName);
  const REG2 = taskName && /^[A-Za-z\d]/.test(taskName);
  const REG3 = taskName && !/-$/.test(taskName);
  const REG4 = taskName && !/\.+\./.test(taskName);
  const REG5 = taskName && !/-+\.$/.test(taskName);
  const REG6 =
    taskName &&
    !/^(?:(?:^|\.)(?:2(?:5[0-5]|[0-4]\d)|1?\d?\d)){4}$/.test(taskName);
  const REG7 = taskName && taskName.length >= 3 && taskName.length <= 20;

  console.info(
    "REG1 && REG2 && REG3 && REG4 && REG5 && REG6 && REG7:",
    REG1,
    REG2,
    REG3,
    REG4,
    REG5,
    REG6,
    REG7
  );

  if (REG1 && REG2 && REG3 && REG4 && REG5 && REG6 && REG7) {
    return true;
  }
  return false;
};
