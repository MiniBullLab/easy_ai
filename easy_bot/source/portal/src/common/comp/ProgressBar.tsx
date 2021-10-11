import * as React from "react";
import PropTypes from "prop-types";
import { makeStyles } from "@material-ui/core/styles";
import { lighten, LinearProgress } from "@material-ui/core";

const BorderLinearProgress = makeStyles((theme) => ({
  root: {
    height: 20,
    backgroundColor: lighten("#b4dbed", 0),
  },
  bar: {
    borderRadius: 0,
    backgroundColor: "#008dc8",
  },
}));

type ProgressProps = {
  value: number;
};

const ProgressBar = (props: ProgressProps): JSX.Element => {
  const classesBorderLinearProgress = BorderLinearProgress();

  const { value } = props;

  return (
    <React.Fragment>
      <LinearProgress
        classes={classesBorderLinearProgress}
        variant="determinate"
        color="secondary"
        value={value || 0}
      />
    </React.Fragment>
  );
};

ProgressBar.propTypes = {
  value: PropTypes.number.isRequired,
};

export default ProgressBar;
