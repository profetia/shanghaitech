# build stage
FROM node:lts-alpine as build-stage
WORKDIR /app
COPY package.json ./
COPY yarn.lock ./
RUN apk add git && \
    git --version && \
    yarn
COPY . .
RUN yarn build

# production stage
FROM nginx:stable-alpine as production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
COPY nginx.conf /etc/nginx/
