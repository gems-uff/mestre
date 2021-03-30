/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package info.heleno.main;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author HelenoCampos
 */
public class ConcatenationType {

    public static void main(String[] args) {
        try {
            classifyConcatenation(args);
        } catch (Exception ex) {
            Logger.getLogger(ConcatenationType.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void classifyConcatenation(String[] args) throws Exception {
        if (args.length == 5) {
            List<String> version1 = Files.readAllLines(new File(args[0]).toPath());
            List<String> version2 = Files.readAllLines(new File(args[1]).toPath());
            List<String> context1 = Files.readAllLines(new File(args[2]).toPath());
            List<String> context2 = Files.readAllLines(new File(args[3]).toPath());
            List<String> solutionContent = Files.readAllLines(new File(args[4]).toPath());

            List<String> solutionClean;
            int solutionCleanBegin = 0, solutionCleanEnd = solutionContent.size();

            boolean passBegin = false, passEnd = false;

            context1 = cleanFormatList(context1);
            version1 = cleanFormatList(version1);
            version2 = cleanFormatList(version2);
            context2 = cleanFormatList(context2);
            solutionContent = cleanFormatList(solutionContent);

            //Finding the range of cleansolution
            for (int i = 0; i < context1.size(); i++) {
                for (int j = 0; j < context1.size(); j++) {
                    if (i >= solutionContent.size() - 1 || j >= solutionContent.size() - 1) {
                        break;
                    } else if (context1.get(i).equals(solutionContent.get(j))) {
                        if (j >= solutionCleanBegin) {
                            solutionCleanBegin = j;
                            passBegin = true;
                        }
                    }
                }
            }

            for (int i = context2.size() - 1; i >= 0; i--) {
                for (int j = solutionContent.size() - 1; j >= solutionContent.size() - context2.size(); j--) {
                    if (i < 0 || j < 0) {
                        break;
                    } else if (context2.get(i).equals(solutionContent.get(j))) {
                        if (j <= solutionCleanEnd) {
                            solutionCleanEnd = j;
                            passEnd = true;
                        }
                    }
                }
            }

            if (solutionCleanBegin > solutionCleanEnd) {
                throw new Exception("Invalid conflicting chunk content!");
            }

            //Cleaning the solution 
            if (solutionCleanBegin == solutionCleanEnd) {
                solutionClean = solutionContent.subList(solutionCleanBegin, solutionCleanEnd);
            } else if (!passBegin && !passEnd) {
                solutionClean = solutionContent;
            } else if (!passBegin) {
                solutionClean = solutionContent.subList(solutionCleanBegin, solutionCleanEnd);
            } else if (!passEnd) {
                solutionClean = solutionContent.subList(solutionCleanBegin + 1, solutionCleanEnd);
            } else if (solutionContent.size() >= solutionCleanEnd + 1) {
                solutionClean = solutionContent.subList(solutionCleanBegin + 1, solutionCleanEnd);
            } else {
                solutionClean = solutionContent.subList(solutionCleanBegin, solutionCleanEnd);
            }

            List<String> aux1 = new ArrayList<>();
            List<String> aux2 = new ArrayList<>();
            List<String> aux3 = new ArrayList<>();
            List<String> aux4 = new ArrayList<>();
            List<String> aux5 = new ArrayList<>();
            List<String> aux6 = new ArrayList<>();
            List<String> aux7 = new ArrayList<>();
            List<String> aux8 = new ArrayList<>();

            //Version 1 + Version 2
            aux1.addAll(version1);
            aux1.addAll(version2);

            //Version 2 + Version 1
            aux2.addAll(version2);
            aux2.addAll(version1);

            //Version 1 + end context 2 + end context 1 + version 2
            if (!context1.isEmpty() && !context2.isEmpty()) {
                aux3.addAll(version1);
                aux3.add(context2.get(0));
                aux3.add(context1.get(context1.size() - 1));
                aux3.addAll(version2);
            }
            //Version 1 + end context 1 + version 2
            if (!context1.isEmpty()) {
                aux4.addAll(version1);
                aux4.add(context1.get(context1.size() - 1));
                aux4.addAll(version2);
            }

            //Version 1 + end context 2 + version 2
            if (!context2.isEmpty()) {
                aux5.addAll(version1);
                aux5.add(context2.get(0));
                aux5.addAll(version2);
            }

            //Version 2 + end context 2 + end context 1 + version 1
            if (!context1.isEmpty() && !context2.isEmpty()) {
                aux6.addAll(version2);
                aux6.add(context2.get(0));
                aux6.add(context1.get(context1.size() - 1));
                aux6.addAll(version1);
            }
            //Version 2 + end context 1 + version 1
            if (!context1.isEmpty()) {
                aux7.addAll(version2);
                aux7.add(context1.get(context1.size() - 1));
                aux7.addAll(version1);
            }
            //Version 2 + end context 2 + version 1
            if (!context2.isEmpty()) {
                aux8.addAll(version2);
                aux8.add(context2.get(0));
                aux8.addAll(version1);
            }
            if (isEqual(aux1, solutionClean)) {
                System.out.println("ConcatenationV1V2");
            } else if (isEqual(aux2, solutionClean)) {
                System.out.println("ConcatenationV2V1");
            } else if (isEqual(aux3, solutionClean)) {
                System.out.println("ConcatenationV1V2");
            } else if (isEqual(aux4, solutionClean)) {
                System.out.println("ConcatenationV1V2");
            } else if (isEqual(aux5, solutionClean)) {
                System.out.println("ConcatenationV1V2");
            } else if (isEqual(aux6, solutionClean)) {
                System.out.println("ConcatenationV2V1");
            } else if (isEqual(aux7, solutionClean)) {
                System.out.println("ConcatenationV2V1");
            } else if (isEqual(aux8, solutionClean)) {
                System.out.println("ConcatenationV2V1");
            } else {
                System.out.println("UnknownConcatenation");
            }
        }else{
            System.out.println("Usage: java -jar classifyConcatenation.jar v1file v2file context1file context2file solutionfile");
        }
    }

    public static boolean isEqual(List<String> version1, List<String> version2) {
        String stringContent1 = listTostring(version1);
        String stringContent2 = listTostring(version2);

        return stringContent1.equals(stringContent2);

    }

    public static String listTostring(List<String> list) {
        StringBuilder result = new StringBuilder();

        for (String line : list) {
            result.append(line);
        }

        return result.toString().replaceAll("\n", "");
    }

    public static String cleanFormat(String line) {

        String lineChanged = line;

        lineChanged = lineChanged.trim();
        lineChanged = lineChanged.replaceAll(" ", "");
        lineChanged = lineChanged.replaceAll("\t", "");

        return lineChanged;
    }

    public static List<String> cleanFormatList(List<String> list) {
        List<String> clone = new ArrayList<>(list);
        List<String> result = new ArrayList<>();

        for (String line : clone) {
            result.add(cleanFormat(line));
        }

        return result;
    }
}
