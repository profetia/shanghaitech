/**
 * main.js
 *
 * Bootstraps Vuetify and other plugins then mounts the App`
 */

// Components
import App from './App.vue';

// Composables
import { createApp } from 'vue';

// Plugins
import { registerPlugins } from '@/plugins';
import vuetify from './plugins/vuetify';
import './plugins/leaflet';

// Styles
import '../css/leaflet.css';

const app = createApp(App);

registerPlugins(app);

app.use(vuetify).mount('#app');
