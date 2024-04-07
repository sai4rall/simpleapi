package com.example.demo;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Arrays;
import java.util.List;

@RestController
public class MyController {
    @GetMapping("hello")
public List<User> getUsersList(){
        System.out.println("this is called");
    return Arrays.asList(new User("123","ABC","PWD"));
}
    @GetMapping("call2")
public List<User> getUsersList2(){
        System.out.println("this is called");
    return Arrays.asList(new User("333","ABC","admin"));
}
}
