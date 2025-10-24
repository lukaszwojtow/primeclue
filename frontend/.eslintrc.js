module.exports = {
  root: true,
  env: {
    node: true,
  },
  extends: [
    'plugin:vue/vue3-essential',
    'eslint:recommended',
  ],
  parserOptions: {
    ecmaVersion: 2020,
  },
  rules: {
	'vue/no-deprecated-v-on-native-modifier': 'off',
	'vue/multi-word-component-names': 'off',
  },
};

