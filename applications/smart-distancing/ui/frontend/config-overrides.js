// https://material-ui.com/guides/minimizing-bundle-size/#option-2
/* config-overrides.js */
const {useBabelRc, override, addWebpackExternals} = require('customize-cra');

module.exports = override(
    useBabelRc(),
    addWebpackExternals({
      react: "React",
      "react-dom": "ReactDOM"
    }),
);
