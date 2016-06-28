package com.gmo.isto.dlwork.tools;

/**
 * Created by usr0101862 on 2016/06/05.
 */
public class DocItem {
    private String docId;
    private String docContent;

    public DocItem(String docId, String docContent) {
        this.docId = docId;
        this.docContent = docContent;
    }


    public String getDocId() {
        return docId;
    }

    public void setDocId(String docId) {
        this.docId = docId;
    }

    public String getDocContent() {
        return docContent;
    }

    public void setDocContent(String docContent) {
        this.docContent = docContent;
    }


}
