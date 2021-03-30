The goal of this Java project is to relabel a conflict scenario which was labelled as *Concatenation* in the database into *ConcatenationV1V2* or *ConcatenationV2V1*. It uses the following information: V1, V2, context, and resolution content. This program is used during the relabelling of the conflicts database.

The labelling algorithm was extracted from https://github.com/gems-uff/merge-nature.

## How to build

Simply execute the maven goal.

```bash
$ mvn install
```

The executable jar file will be created in the target folder with the name *classifyConcatenation.jar*.

## How to execute

To label a merge scenario, use the following command.

```bash
$ java -jar classifyConcatenation.jar v1 v2 context1 context2 solution
```

Where v1,v2, context1, context2, and solution are paths to files containing the respective text for each one.

The result will be printed to the standard output (stdout).

---

Consider the following conflict scenario as an example:

- Project: www.github.com/jtalks-org/jcommune
- Merge SHA: ea95f4e815b464332727b3a6e4d0e51f06dbe9a8
    - Left: af1452de76f0aac34ef8eb30956b9c26edcbecae
    - Right: 29a02308ca5557b393a44605cc9cba752a88acc7
    - Base: 1e1903491b431b96195123ea92eee60180a1e678
- File: TransactionalTopicService.java


Scenario:
``` java
    private BranchDao branchDao;
    private NotificationService notificationService;
    private SubscriptionService subscriptionService;
<<<<<<< HEAD
    private PaginationService paginationService;
=======
    private UserService userService;
>>>>>>> 29a02308ca5557b393a44605cc9cba752a88acc7

    /**
     * Create an instance of User entity based service
```

In this example, the following files and its contents will be considered.

v1:
``` java
    private PaginationService paginationService;
```

v2:
``` java
    private UserService userService;
```

context1:
``` java
    private BranchDao branchDao;
    private NotificationService notificationService;
    private SubscriptionService subscriptionService;
```

context2:
``` java

    /**
     * Create an instance of User entity based service
```

solution:
``` java
    private BranchDao branchDao;
    private NotificationService notificationService;
    private SubscriptionService subscriptionService;
    private PaginationService paginationService;
    private UserService userService;

    /**
     * Create an instance of User entity based service
```

If these files are provided to the executable jar, it will be classified as *ConcatenationV1V2*.