// https://material-ui.com/guides/minimizing-bundle-size/#option-2
/* config-overrides.js */
const {useBabelRc, override} = require('customize-cra')

module.exports = override(
    useBabelRc()
);
