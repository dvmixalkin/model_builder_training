# Как этим пользоваться:
- git pull "этот репозиторий"
- git remote add all http://cloudtasks.acti.ru:41280/idk/model_builder_training.git
- git remote set-url --add --push all http://cloudtasks.acti.ru:41280/idk/model_builder_training.git
- git remote set-url --add --push all http://10.200.0.56/idk/model_builder_training.git
- ...
- git push all master

git pull не работает, используйте git fetch;

- git fetch --all

Подробнее как работать с 2 ремоутами одновременно можно узнать тут: https://jigarius.com/blog/multiple-git-remote-repositories